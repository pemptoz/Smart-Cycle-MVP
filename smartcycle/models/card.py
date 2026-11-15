"""Final model bundle (models/model.joblib) and model card (models/model_card.json)."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import joblib
import sklearn

from smartcycle.features.preprocess import PipelineConfig

CARD_LIMITATIONS = [
    "Window-level classifier of 4 plan classes only (Smooth, Rough, Pothole, Irregularity).",
    "Stationary windows (speed < threshold) are not classified.",
    "Severity (if shown elsewhere) is an uncalibrated heuristic, not produced by this model.",
    "Metrics come from ride-grouped CV on the training data; small datasets give wide error "
    "bars. Synthetic results say nothing about real roads.",
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def cv_summary(results: dict) -> dict:
    out = {}
    for n, r in results.items():
        s = r.summary
        out[n] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                  "macro_f1_folds": s["macro_f1_folds"],
                  "f1_per_class": {c: v["pooled"]["f1"] for c, v in s["per_class"].items()},
                  "event_recall": {c: s["events"][c]["event_recall"]
                                   for c in ("Pothole", "Irregularity")},
                  "false_alarms_per_km": {c: s["events"][c]["false_alarms_per_km"]
                                          for c in ("Pothole", "Irregularity")}}
    return out


def save_model(out_dir: str | Path, model, model_name: str, cfg: PipelineConfig,
               feature_columns: list[str], ds_sum: dict, results: dict, cv_mode: str,
               data_dir: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "model_name": model_name, "config": cfg.to_dict(),
              "feature_columns": list(feature_columns), "classes": list(model.classes_),
              "data_provenance": ds_sum["data_provenance"]}
    mpath, cpath = out / "model.joblib", out / "model_card.json"
    joblib.dump(bundle, mpath)
    card = {
        "model": model_name, "data_provenance": ds_sum["data_provenance"],
        "n_rides": ds_sum["n_rides"], "n_windows": ds_sum["n_windows"],
        "n_windows_per_class": ds_sum["n_windows_per_class"],
        "n_stationary_windows_dropped": ds_sum["n_stationary_windows_dropped"],
        "feature_set": cfg.feature_set, "include_speed": cfg.include_speed,
        "n_features": len(feature_columns), "fs": cfg.fs, "window_s": cfg.window_s,
        "step_s": cfg.step_s, "stationary_mps": cfg.stationary_mps,
        "classes": list(model.classes_), "cv_mode": cv_mode, "cv_metrics": cv_summary(results),
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__, "git_commit": git_commit(),
        "data_dir": str(data_dir), "limitations": CARD_LIMITATIONS,
    }
    if ds_sum["data_provenance"] != "real":
        card["warning"] = "Trained on SYNTHETIC data: do not report these metrics as real."
    cpath.write_text(json.dumps(card, indent=1))
    return mpath, cpath


def load_model(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `python -m smartcycle train` first")
    return joblib.load(p)
