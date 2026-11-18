"""Load the committed artifacts; compute the extra local-only artifacts (in-sample metrics of
the SAVED model, dataset description) when the real data is available.

Nothing here retrains or edits the model or its card."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from smartcycle.io.labels import LABELS

ARTIFACTS = {
    "metrics": "reports/metrics.json",
    "aug": "reports/aug/metrics.json",
    "calib": "reports/calib/metrics.json",
    "card": "models/model_card.json",
    "notes": "models/model_notes.json",
    "insample": "reports/insample.json",
    "dataset": "reports/dataset.json",
    "example": "reports/walkthrough_example.json",
}


def load_artifacts(root: str | Path = ".") -> dict:
    """{name: parsed JSON or None if the file is missing}."""
    root = Path(root)
    out = {}
    for k, rel in ARTIFACTS.items():
        p = root / rel
        out[k] = json.loads(p.read_text()) if p.exists() else None
    return out


def compute_insample(data_dir: str | Path, model_path: str | Path, card: dict) -> dict:
    """Metrics of the saved model on its OWN training windows (same data, mount filter and
    config as the card). An optimistic upper bound, used only to show the overfitting gap."""
    from sklearn.metrics import accuracy_score, f1_score

    from smartcycle.features.build import build_windows
    from smartcycle.features.preprocess import PipelineConfig
    from smartcycle.io.dataset import load_dataset, select_mount
    from smartcycle.models.card import load_model

    bundle = load_model(model_path)
    cfg = PipelineConfig.from_dict(bundle["config"])
    ds = select_mount(load_dataset(data_dir), card.get("mount_filter") or "all")
    wt = build_windows(ds, cfg)
    y = wt.meta["label"].to_numpy()
    pred = bundle["model"].predict(wt.X[bundle["feature_columns"]])
    per = f1_score(y, pred, labels=list(LABELS), average=None, zero_division=0)
    return {
        "what": "in-sample (training windows of the saved model); optimistic, NOT an evaluation",
        "model": bundle.get("model_name"), "data_provenance": card.get("data_provenance"),
        "n_windows": int(len(y)), "matches_card_n_windows": int(len(y)) == card.get("n_windows"),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=list(LABELS), average="macro", zero_division=0)),
        "f1_per_class": {c: float(v) for c, v in zip(LABELS, per)},
    }


def dataset_info(data_dir: str | Path) -> dict:
    """Per-ride description of a dataset (no GPS coordinates, no device ids)."""
    from smartcycle.io.csvsniff import read_csv_any
    from smartcycle.io.dataset import read_provenance
    from smartcycle.io.labels import load_labels
    from smartcycle.io.loader import load_ride

    root = Path(data_dir)
    meta = read_csv_any(root / "rides.csv")
    meta.columns = [c.strip().lower() for c in meta.columns]
    labels = load_labels(root / "labels.csv") if (root / "labels.csv").exists() else None
    rides = []
    for _, r in meta.iterrows():
        df = load_ride(root / str(r["file"]))
        dt = np.diff(df["timestamp_ms"].to_numpy(dtype=float))
        n_fix = int(df["latitude"].notna().sum())
        ev = {}
        if labels is not None:
            sub = labels[labels["ride_id"] == r["ride_id"]]
            ev = {c: int((sub["label"] == c).sum()) for c in LABELS if (sub["label"] == c).any()}
        rides.append({"ride_id": str(r["ride_id"]), "mount": str(r["mount"]),
                      "notes": str(r.get("notes") or ""),
                      "duration_s": round(float(df["t_s"].iloc[-1]), 1),
                      "n_samples": int(len(df)),
                      "fs_hz": round(1000.0 / float(np.median(dt)), 1) if len(dt) else None,
                      "n_gps_fixes": n_fix, "labelled_intervals": ev})
    return {"data_dir": str(root), "provenance": read_provenance(root), "rides": rides,
            "n_rides": len(rides),
            "total_duration_s": round(sum(x["duration_s"] for x in rides), 1),
            "mounts": {m: sum(x["mount"] == m for x in rides)
                       for m in sorted({x["mount"] for x in rides})}}


def compute_example(data_dir: str | Path, model_path: str | Path, network_path: str | Path,
                    ride_id: str = "bump", contrast_ride: str = "smooth") -> dict:
    """One concrete trace through the pipeline for the study guide: the strongest Pothole event
    of `ride_id` (window features, class probabilities, event, severity, map matching) and a
    typical Smooth window of `contrast_ride`. No coordinates are stored (privacy)."""
    from smartcycle.features.preprocess import PipelineConfig, haversine_m, preprocess_ride
    from smartcycle.features.build import ride_windows
    from smartcycle.geo.network import load_network
    from smartcycle.io.loader import load_ride
    from smartcycle.models.card import load_model
    from smartcycle.pipeline import process_ride

    root = Path(data_dir)
    bundle = load_model(model_path)
    cfg = PipelineConfig.from_dict(bundle["config"])
    net = load_network(Path(network_path))

    def window_table(rid: str):
        df = load_ride(root / "canonical" / f"{rid}.csv")
        meta, X = ride_windows(preprocess_ride(df, cfg, rid), cfg, labelled=False)
        P = bundle["model"].predict_proba(X[bundle["feature_columns"]])
        return df, meta, X, P

    df, meta, X, P = window_table(ride_id)
    res = process_ride(df, bundle, net, mount="handlebar")
    pots = [e for e in res.events if e["type"] == "Pothole"]
    if not pots:
        raise ValueError(f"{ride_id}: no Pothole event")
    ev = max(pots, key=lambda e: e["peak"] or 0)
    i = int(np.argmin(np.abs(meta["t_peak"].to_numpy() - ev["t_peak_ms"] / 1000)))
    classes = list(bundle["classes"])
    seg = net.props[net.index[ev["segment_id"]]] if ev["segment_id"] else {}
    snap = (float(haversine_m(ev["lat"], ev["lon"], ev["snapped_lat"], ev["snapped_lon"]))
            if ev["matched"] else None)
    feat = ["a_vert__max", "a_vert__min", "a_vert__rms", "a_vert__std", "a_hmag__max",
            "a_dev__max", "a_vert__kurt", "a_vert__dom_freq"]
    win = {"t_start_s": float(meta["t_start"].iloc[i]), "t_end_s": float(meta["t_end"].iloc[i]),
           "n_samples": int(round(cfg.window_s * cfg.fs)), "speed_mps": float(meta["speed"].iloc[i]),
           "features": {f: float(X[f].iloc[i]) for f in feat if f in X.columns},
           "proba": {c: float(P[i, j]) for j, c in enumerate(classes)},
           "pred": classes[int(P[i].argmax())]}
    _, cm, cX, cP = window_table(contrast_ride)
    k = int(np.argsort(cX["a_vert__rms"].to_numpy())[len(cX) // 2])  # median-rms window
    contrast = {"ride_id": contrast_ride,
                "features": {f: float(cX[f].iloc[k]) for f in feat if f in cX.columns},
                "proba": {c: float(cP[k, j]) for j, c in enumerate(classes)},
                "pred": classes[int(cP[k].argmax())]}
    return {"what": "worked example for docs/WALKTHROUGH.md (training ride: in-sample)",
            "ride_id": ride_id, "n_features": len(bundle["feature_columns"]),
            "window": win, "contrast_window": contrast,
            "event": {"type": ev["type"], "t_start_s": ev["t_start_ms"] / 1000,
                      "t_end_s": ev["t_end_ms"] / 1000, "peak_mps2": ev["peak"],
                      "severity": ev["severity"], "confidence": ev["confidence"],
                      "matched": ev["matched"], "snap_distance_m": snap,
                      "segment_highway": seg.get("highway"),
                      "segment_is_cycleway": seg.get("is_cycleway"),
                      "segment_length_m": seg.get("length_m")},
            "ride": {"n_events": len(res.events),
                     "events_by_type": {t: sum(e["type"] == t for e in res.events)
                                        for t in ("Pothole", "Irregularity", "Rough")},
                     "pct_matched": res.summary["gps"].get("pct_matched")}}
