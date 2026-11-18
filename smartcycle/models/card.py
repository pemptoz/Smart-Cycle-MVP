"""Final model bundle (models/model.joblib) and model card (models/model_card.json)."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import joblib
import sklearn

from smartcycle.features.preprocess import PipelineConfig

CARD_LIMITATIONS = [
    "Window-level classifier of 4 plan classes only (Smooth, Rough, Pothole, Irregularity).",
    "Stationary windows (speed < threshold) are not classified.",
    "Severity (if shown elsewhere) is an uncalibrated heuristic, not produced by this model.",
    "Metrics come from cross-validation on the training data (see cv_protocol); small datasets "
    "give wide error bars. Blocked CV inside rides is optimistic and supports no ride-level "
    "generalisation claim. Synthetic results say nothing about real roads.",
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    """True when the working tree has uncommitted changes (tracked or untracked, not ignored)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True,
                             timeout=5, check=True, cwd=Path(__file__).resolve().parent).stdout
        return bool(out.strip())
    except Exception:
        return None


PROVENANCE_WARNINGS = {
    "synthetic": "Trained on SYNTHETIC data: do not report these metrics as real.",
    "mixed": "Trained on REAL data plus augmented and/or synthetic windows (MIXED). Test folds "
             "contained real windows only.",
}


def cv_summary(results: dict) -> dict:
    out = {}
    for n, r in results.items():
        s = r.summary
        out[n] = {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                  "macro_f1_folds": s["macro_f1_folds"],
                  "f1_per_class": {c: v["pooled"]["f1"] for c, v in s["per_class"].items()},
                  "events_detected": {c: s["events"][c]["detected_of"]
                                      for c in ("Pothole", "Irregularity")},
                  "false_alarms": {c: s["events"][c]["false_alarms"]
                                   for c in ("Pothole", "Irregularity")},
                  "false_alarms_per_km": {c: s["events"][c]["false_alarms_per_km"]
                                          for c in ("Pothole", "Irregularity")},
                  "events_predicted": {c: s["events"][c].get("n_predicted")
                                       for c in ("Pothole", "Irregularity")},
                  "event_precision": {c: s["events"][c].get("event_precision")
                                      for c in ("Pothole", "Irregularity")},
                  "event_matching": s["events"].get("matching")}
    return out


def save_model(out_dir: str | Path, model, model_name: str, cfg: PipelineConfig,
               feature_columns: list[str], ds_sum: dict, results: dict, cv_mode: str,
               data_dir: str, cv_protocol: dict | None = None,
               augmentation: dict | None = None, n_synthetic_windows: int = 0,
               n_augmented_windows: int = 0) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirty = git_dirty()
    if dirty:
        print("WARNING: training with a DIRTY git tree: git_commit does not identify the code "
              "exactly (model_card.json records git_dirty=true). Commit first for a "
              "reproducible model.", file=sys.stderr)
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
        "cv_protocol": cv_protocol or {"mode": cv_mode},
        "augmentation": augmentation, "n_augmented_windows": int(n_augmented_windows),
        "n_synthetic_windows": int(n_synthetic_windows),
        "sklearn_version": sklearn.__version__, "git_commit": git_commit(),
        "git_dirty": dirty, "data_dir": str(data_dir), "limitations": CARD_LIMITATIONS,
    }
    if ds_sum.get("mount_filter"):
        card["mount_filter"] = ds_sum["mount_filter"]
    if ds_sum["data_provenance"] in PROVENANCE_WARNINGS:
        card["warning"] = PROVENANCE_WARNINGS[ds_sum["data_provenance"]]
    cpath.write_text(json.dumps(card, indent=1))
    return mpath, cpath


REPRODUCE_HINT = ("retrain with the Reproduce command (README, section 'Reproduce the real "
                  "results'): python -m smartcycle train --data-dir data/real_public --mount "
                  "handlebar --cv blocked --no-speed --ablate --final-model rf")


class ModelLoadError(RuntimeError):
    """The model file is missing, corrupt, or was saved by another scikit-learn version."""


def load_model(path: str | Path, card_path: str | Path | None = None) -> dict:
    """Load a model bundle, refusing anything that would silently mis-predict: a missing or
    corrupt file, a pickle from another scikit-learn version (InconsistentVersionWarning is
    an error here), or a model card whose `sklearn_version` differs from the installed one.
    `card_path` defaults to `model_card.json` next to the model."""
    import warnings

    from sklearn.exceptions import InconsistentVersionWarning

    p = Path(path)
    if not p.exists():
        raise ModelLoadError(f"{p} not found: run `python -m smartcycle train` first, or "
                             f"{REPRODUCE_HINT}")
    cp = Path(card_path) if card_path is not None else p.parent / "model_card.json"
    if cp.exists():
        try:
            want = json.loads(cp.read_text()).get("sklearn_version")
        except (json.JSONDecodeError, OSError, AttributeError):
            want = None
        if want and want != sklearn.__version__:
            raise ModelLoadError(f"{p} was trained with scikit-learn {want}, installed "
                                 f"{sklearn.__version__}: install scikit-learn=={want} or "
                                 f"{REPRODUCE_HINT}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            bundle = joblib.load(p)
    except Exception as e:  # noqa: BLE001 - any unpickling failure means "not loadable"
        raise ModelLoadError(f"{p} could not be loaded ({type(e).__name__}: {e}): "
                             f"{REPRODUCE_HINT}") from None
    if not isinstance(bundle, dict) or not {"model", "feature_columns", "classes",
                                            "config"} <= set(bundle):
        raise ModelLoadError(f"{p} is not a Smart Cycle model bundle: {REPRODUCE_HINT}")
    return bundle
