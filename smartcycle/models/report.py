"""Report artifacts in reports/: metrics.json, metrics.md, confusion_matrix.png,
feature_importance.png, signal_examples.png, per_ride.csv, ablations.csv.
Every artifact states the data provenance and the number of rides / windows per class."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from smartcycle.features.build import WindowTable  # noqa: E402
from smartcycle.features.preprocess import PipelineConfig  # noqa: E402
from smartcycle.io.labels import LABELS  # noqa: E402
from smartcycle.models.evaluation import ModelResult, Splits  # noqa: E402
from smartcycle.models.zoo import MODEL_LABELS  # noqa: E402


def provenance_line(ds_sum: dict) -> str:
    prov = ds_sum["data_provenance"].upper()
    warn = " — SYNTHETIC DATA, NOT REAL RESULTS" if ds_sum["data_provenance"] != "real" else ""
    counts = ", ".join(f"{k} {v}" for k, v in ds_sum["n_windows_per_class"].items())
    return (f"Data provenance: {prov}{warn} | {ds_sum['n_rides']} rides, "
            f"{ds_sum['n_windows']} windows ({counts})")


def _fmt(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _pm(d: dict) -> str:
    return "–" if d["mean"] is None else f"{d['mean']:.3f} ± {d['std']:.3f}"


def metrics_markdown(results: dict[str, ModelResult], ds_sum: dict, sp: Splits,
                     cfg: PipelineConfig, ablations: pd.DataFrame | None) -> str:
    L = [f"# Smart Cycle — classifier evaluation", "", f"**{provenance_line(ds_sum)}**", ""]
    if sp.mode == "blocked":
        from smartcycle.models.evaluation import BLOCKED_HEADER
        pr = sp.protocol
        L[2:2] = [f"> **{BLOCKED_HEADER}.**", ""]
        L.append(f"- Blocked CV: blocks of {pr['block_s']:g} s inside each ride "
                 f"({pr['n_blocks']} blocks; per class {pr['blocks_per_class']}), "
                 f"StratifiedGroupKFold over blocks, k = {pr['n_splits']}, split seed "
                 f"{pr['seed']} (first seed giving every class in every test fold), training "
                 f"windows within {pr['purge_s']:g} s of a test window removed (purge).")
        for f in pr["folds"]:
            L.append(f"  - fold {f['fold']}: train {f['n_train']} (purged {f['n_purged']}), "
                     f"test {f['n_test']}, classes {f['test_classes']}"
                     + (f", MISSING {f['missing_test_classes']}" if f["missing_test_classes"]
                        else ""))
    L.append(f"- CV: {sp.mode}, {len(sp.folds)} folds"
             + (" (grouped by ride: no ride in both train and test)" if sp.mode == "grouped"
                else ""))
    if sp.warning:
        L.append(f"- **WARNING:** {sp.warning}")
    L += [f"- Windows: {cfg.window_s} s, step {cfg.step_s} s, resampled to {cfg.fs:g} Hz; "
          f"features: `{cfg.feature_set}`, speed features: {cfg.include_speed}",
          f"- Stationary windows (< {cfg.stationary_mps} m/s) dropped: "
          f"{ds_sum['n_stationary_windows_dropped']}",
          "- Pooled = metrics on all out-of-fold predictions; mean ± std = across folds "
          "(classes absent from a test fold are skipped for that fold).",
          "- Picking the best of 3 models on the same CV is slightly optimistic.", "",
          "## Overall", "", "| Model | Accuracy (pooled) | Macro-F1 (pooled) | Macro-F1 (folds) |",
          "|---|---|---|---|"]
    for n, r in results.items():
        s = r.summary
        L.append(f"| {MODEL_LABELS[n]} | {_fmt(s['accuracy'])} | {_fmt(s['macro_f1'])} | "
                 f"{_pm(s['macro_f1_folds'])} |")
    L += ["", "## Per class (F1: pooled / folds mean ± std)", "",
          "| Model | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for n, r in results.items():
        pc = r.summary["per_class"]
        L.append(f"| {MODEL_LABELS[n]} | " + " | ".join(
            f"{_fmt(pc[c]['pooled']['f1'])} / {_pm(pc[c]['f1'])}" for c in LABELS) + " |")
    L += ["", "## Per class precision / recall (pooled)", "",
          "| Model | " + " | ".join(f"{c} P / R" for c in LABELS) + " |",
          "|---|" + "---|" * len(LABELS)]
    for n, r in results.items():
        pc = r.summary["per_class"]
        L.append(f"| {MODEL_LABELS[n]} | " + " | ".join(
            f"{_fmt(pc[c]['pooled']['precision'])} / {_fmt(pc[c]['pooled']['recall'])}"
            for c in LABELS) + " |")
    ev0 = next(iter(results.values())).summary["events"]
    L += ["", f"## Event level (tolerance ±{ev0['tolerance_s']:g} s, evaluated distance "
          f"{ev0['distance_km']:.2f} km)", "",
          "| Model | Pothole recall | Pothole false alarms/km | Irregularity recall | "
          "Irregularity false alarms/km |", "|---|---|---|---|---|"]
    for n, r in results.items():
        e = r.summary["events"]
        L.append(f"| {MODEL_LABELS[n]} | {_fmt(e['Pothole']['event_recall'])} "
                 f"({e['Pothole']['detected']}/{e['Pothole']['n_labelled']}) | "
                 f"{_fmt(e['Pothole']['false_alarms_per_km'], 2)} | "
                 f"{_fmt(e['Irregularity']['event_recall'])} "
                 f"({e['Irregularity']['detected']}/{e['Irregularity']['n_labelled']}) | "
                 f"{_fmt(e['Irregularity']['false_alarms_per_km'], 2)} |")
    L += ["", "## Per ride (out-of-fold): windows / recall per class, macro-F1", ""]
    for n, r in results.items():
        L += [f"**{MODEL_LABELS[n]}**", "",
              "| Ride | " + " | ".join(f"{c} n / R" for c in LABELS) + " | Macro-F1 |",
              "|---|" + "---|" * (len(LABELS) + 1)]
        for rid, d in r.summary["per_ride"].items():
            L.append(f"| {rid} | " + " | ".join(
                f"{d['n'][c]} / {_fmt(d['recall'][c])}" if d["n"][c] else "–"
                for c in LABELS) + f" | {_fmt(d['macro_f1'])} |")
        L.append("")
    L += ["## Confusion matrices (pooled out-of-fold, rows = true, row-normalised)", ""]
    for n, r in results.items():
        cm = r.summary["confusion"]
        L += [f"**{MODEL_LABELS[n]}**", "", "| true \\ pred | " + " | ".join(LABELS) + " |",
              "|---|" + "---|" * len(LABELS)]
        for i, c in enumerate(LABELS):
            L.append(f"| {c} | " + " | ".join(
                f"{cm['row_normalised'][i][j]:.2f} ({cm['counts'][i][j]})"
                for j in range(len(LABELS))) + " |")
        L.append("")
    if ablations is not None and len(ablations):
        L += [f"## Experiments / ablations (RandomForest, same CV protocol: {sp.mode})", "",
              "| Experiment | Windows | Macro-F1 (pooled) | Macro-F1 (folds) | "
              + " | ".join(f"F1 {c}" for c in LABELS)
              + " | Pothole ev. recall | Irregularity ev. recall |",
              "|---|---|---|---|" + "---|" * len(LABELS) + "---|---|"]
        for _, a in ablations.iterrows():
            fm = a.get("macro_f1_fold_mean")
            folds = ("–" if fm is None or pd.isna(fm)
                     else f"{fm:.3f} ± {a['macro_f1_fold_std']:.3f}")
            L.append(f"| {a['ablation']} | {a['n_windows']} | {_fmt(a['macro_f1'])} | {folds} | "
                     + " | ".join(_fmt(_num(a.get(f"f1_{c}"))) for c in LABELS)
                     + f" | {_fmt(_num(a.get('pothole_event_recall')))} | "
                       f"{_fmt(_num(a.get('irregularity_event_recall')))} |")
        L.append("")
        spd = ablations.attrs.get("speed_by_class") or []
        if spd:
            L += ["### Speed per class (moving windows, m/s)", "",
                  "| Class | n | mean | median | std |", "|---|---|---|---|---|"]
            for r in spd:
                L.append(f"| {r['label']} | {r['n']} | {_fmt(_num(r['mean_mps']), 2)} | "
                         f"{_fmt(_num(r['median_mps']), 2)} | {_fmt(_num(r['std_mps']), 2)} |")
            L.append("")
        for k, res in (ablations.attrs.get("pocket") or {}).items():
            L += [f"### (iv) Orientation test, train handlebar -> test pocket (`{k}`)", "",
                  f"Classes labelled in pocket (the only ones scored): "
                  f"{', '.join(res['classes_in_pocket'])}; {res['n_windows']} windows; "
                  f"macro-F1 over these classes {_fmt(res['macro_f1'])}; predicted counts "
                  f"{res['predicted']}.", "", "| Class | Precision | Recall | F1 | n |",
                  "|---|---|---|---|---|"]
            for c, v in res["per_class"].items():
                L.append(f"| {c} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | "
                         f"{_fmt(v['f1'])} | {v['support']} |")
            L.append("")
        for n in ablations.attrs.get("notes") or []:
            L += [f"> **{n}**", ""]
    L += ["## Windows per class", "", "| " + " | ".join(LABELS) + " |",
          "|" + "---|" * len(LABELS),
          "| " + " | ".join(str(ds_sum["n_windows_per_class"][c]) for c in LABELS) + " |", "",
          f"_{provenance_line(ds_sum)}_", ""]
    return "\n".join(L)


def _footer(fig, ds_sum: dict) -> None:
    fig.text(0.5, 0.005, provenance_line(ds_sum), ha="center", va="bottom", fontsize=8,
             color="#c93b27" if ds_sum["data_provenance"] != "real" else "#333")


def plot_confusion(results: dict[str, ModelResult], ds_sum: dict, path: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8), squeeze=False)
    for j, (name, r) in enumerate(results.items()):
        cm = r.summary["confusion"]
        for i, (key, title) in enumerate((("counts", "counts"),
                                           ("row_normalised", "row-normalised"))):
            M = np.array(cm[key], dtype=float)
            ax = axes[i, j]
            ax.imshow(M if key == "row_normalised" else M / max(M.max(), 1), cmap="Greens",
                      vmin=0, vmax=1)
            for a in range(len(LABELS)):
                for b in range(len(LABELS)):
                    v = M[a, b]
                    txt = f"{int(v)}" if key == "counts" else f"{v:.2f}"
                    dark = (v / max(M.max(), 1) if key == "counts" else v) > 0.6
                    ax.text(b, a, txt, ha="center", va="center", fontsize=8,
                            color="white" if dark else "black")
            ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, fontsize=8)
            ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
            ax.set_title(f"{MODEL_LABELS[name]} ({title})", fontsize=9)
            ax.set_xlabel("predicted", fontsize=8)
            ax.set_ylabel("true", fontsize=8)
    fig.suptitle("Pooled out-of-fold confusion matrices (ride-grouped CV)")
    _footer(fig, ds_sum)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(r: ModelResult, ds_sum: dict, path: Path, top: int = 15) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, s, title in ((axes[0], r.impurity_importance, "RF impurity importance (mean of folds)"),
                         (axes[1], r.permutation_importance,
                          "Permutation importance on held-out folds (Δ macro-F1)")):
        if s is None:
            ax.set_axis_off()
            continue
        s = s.sort_values(ascending=True).tail(top)
        ax.barh(range(len(s)), s.to_numpy(), color="#3f9a4a")
        ax.set_yticks(range(len(s)), s.index, fontsize=8)
        ax.set_title(title, fontsize=10)
    _footer(fig, ds_sum)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_signal_examples(wt: WindowTable, ds_sum: dict, path: Path) -> None:
    """One window per class (the one with median peak magnitude), a_vert and a_hmag."""
    fig, axes = plt.subplots(1, len(LABELS), figsize=(4 * len(LABELS), 3.4), sharey=True)
    for ax, c in zip(axes, LABELS):
        sub = wt.meta[wt.meta["label"] == c]
        if sub.empty:
            ax.set_title(f"{c} (no window)")
            continue
        row = sub.iloc[(sub["peak_mag"] - sub["peak_mag"].median()).abs().argmin()]
        prep = wt.preps[row["ride_id"]]
        m = (prep.t >= row["t_start"] - 1.0) & (prep.t < row["t_end"] + 1.0)
        ax.plot(prep.t[m] - row["t_start"], prep.signals["a_vert"][m], lw=0.8, label="a_vert")
        ax.plot(prep.t[m] - row["t_start"], prep.signals["a_hmag"][m], lw=0.8, label="a_hmag")
        ax.axvspan(0, row["t_end"] - row["t_start"], color="#bfdbab", alpha=0.4)
        ax.set_title(f"{c} — {row['ride_id']} ({row['mount']})", fontsize=9)
        ax.set_xlabel("s (window shaded, ±1 s context)", fontsize=8)
    axes[0].set_ylabel("m/s²")
    axes[0].legend(fontsize=7)
    _footer(fig, ds_sum)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_reports(out_dir: str | Path, results: dict[str, ModelResult], wt: WindowTable,
                  ds_sum: dict, sp: Splits, cfg: PipelineConfig,
                  ablations: pd.DataFrame | None = None) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {**ds_sum, "provenance_note": provenance_line(ds_sum), "cv_mode": sp.mode,
               "cv_protocol": sp.protocol, "cv_warning": sp.warning, "n_folds": len(sp.folds), "config": cfg.to_dict(),
               "models": {n: {**r.summary, "folds": r.fold_metrics} for n, r in results.items()}}
    if ablations is not None:
        payload["ablations"] = ablations.astype(object).where(ablations.notna(), None) \
            .to_dict(orient="records")
        payload["speed_by_class"] = ablations.attrs.get("speed_by_class")
        payload["pocket_test"] = ablations.attrs.get("pocket")
        payload["experiment_notes"] = ablations.attrs.get("notes")
    paths = [out / "metrics.json", out / "metrics.md", out / "per_ride.csv",
             out / "confusion_matrix.png", out / "signal_examples.png"]
    paths[0].write_text(json.dumps(payload, indent=1, default=_json_default))
    paths[1].write_text(metrics_markdown(results, ds_sum, sp, cfg, ablations))
    per_ride = wt.meta.groupby("ride_id").agg(mount=("mount", "first"),
                                              n_windows=("label", "size"))
    for n, r in results.items():
        per_ride[f"macro_f1_{n}"] = pd.Series(r.summary["per_ride_macro_f1"])
    per_ride["data_provenance"] = ds_sum["data_provenance"]
    per_ride.to_csv(paths[2])
    plot_confusion(results, ds_sum, paths[3])
    plot_signal_examples(wt, ds_sum, paths[4])
    if "rf" in results and results["rf"].impurity_importance is not None:
        paths.append(out / "feature_importance.png")
        plot_importance(results["rf"], ds_sum, paths[-1])
    if ablations is not None:
        paths.append(out / "ablations.csv")
        ablations.assign(data_provenance=ds_sum["data_provenance"]).to_csv(paths[-1], index=False)
    return paths


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
