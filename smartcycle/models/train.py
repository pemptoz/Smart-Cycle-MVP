"""`evaluate` and `train` orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from smartcycle.features.build import WindowTable, build_windows
from smartcycle.features.features import config_columns
from smartcycle.features.preprocess import PipelineConfig
from smartcycle.io.dataset import Dataset, load_dataset
from smartcycle.models.card import save_model
from smartcycle.models.evaluation import ModelResult, Splits, cross_validate, dataset_summary, \
    run_ablations
from smartcycle.models.report import provenance_line, write_reports
from smartcycle.models.zoo import MODEL_LABELS, MODEL_NAMES, make_model


@dataclass
class EvalRun:
    ds: Dataset
    wt: WindowTable
    results: dict[str, ModelResult]
    splits: Splits
    columns: list[str]
    ds_sum: dict
    ablations: pd.DataFrame | None
    artifacts: list[Path]


def evaluate(data_dir: str | Path, cfg: PipelineConfig, reports_dir: str | Path = "reports",
             ablate: bool = False, importance: bool = True) -> EvalRun:
    ds = load_dataset(data_dir)
    wt = build_windows(ds, cfg)
    ds_sum = dataset_summary(ds, wt)
    print(provenance_line(ds_sum))
    results, sp, cols = cross_validate(wt, cfg, MODEL_NAMES, importance=importance)
    abl = run_ablations(ds, cfg, wt, results.get("rf")) if ablate else None
    paths = write_reports(reports_dir, results, wt, ds_sum, sp, cfg, abl)
    return EvalRun(ds, wt, results, sp, cols, ds_sum, abl, paths)


def results_table(run: EvalRun) -> str:
    rows = []
    for n, r in run.results.items():
        s = r.summary
        e = s["events"]
        rows.append({"model": MODEL_LABELS[n], "accuracy": s["accuracy"],
                     "macro_f1": s["macro_f1"],
                     "macro_f1_folds": f"{s['macro_f1_folds']['mean']:.3f} ± "
                                       f"{s['macro_f1_folds']['std']:.3f}",
                     "pothole_ev_recall": e["Pothole"]["event_recall"],
                     "irreg_ev_recall": e["Irregularity"]["event_recall"],
                     "pothole_FA_km": e["Pothole"]["false_alarms_per_km"],
                     "irreg_FA_km": e["Irregularity"]["false_alarms_per_km"]})
    return pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}")


def pick_final(run: EvalRun, choice: str) -> str:
    if choice == "best":
        return max(run.results, key=lambda n: run.results[n].summary["macro_f1"])
    if choice not in MODEL_NAMES:
        raise ValueError(f"--final-model must be one of {MODEL_NAMES} or 'best'")
    return choice


def train(data_dir: str | Path, cfg: PipelineConfig, out_dir: str | Path = "models",
          reports_dir: str | Path = "reports", final_model: str = "rf",
          ablate: bool = False) -> tuple[EvalRun, str, Path, Path]:
    """CV (all models) + reports, then fit the final model on ALL moving labelled windows."""
    run = evaluate(data_dir, cfg, reports_dir, ablate=ablate)
    name = pick_final(run, final_model)
    cols = config_columns(list(run.wt.X.columns), cfg)
    model = make_model(name).fit(run.wt.X[cols], run.wt.meta["label"].to_numpy())
    mpath, cpath = save_model(out_dir, model, name, cfg, cols, run.ds_sum, run.results,
                              run.splits.mode, str(data_dir))
    return run, name, mpath, cpath
