"""docs/RESULTS.md, generated entirely from the artifacts (see artifacts.ARTIFACTS).
Every number below is read from a JSON file; the caveats are conditions on those numbers."""

from __future__ import annotations

from smartcycle.io.labels import LABELS

MODEL_LABELS = {"rule": "Rule baseline", "rf": "RandomForest", "hgb": "HistGradientBoosting"}
POINT = ("Pothole", "Irregularity")


def f3(x) -> str:
    return "–" if x is None else f"{x:.3f}"


def table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _models_table(m: dict) -> str:
    rows = []
    for k, r in m["models"].items():
        fo = r["macro_f1_folds"]
        rows.append([MODEL_LABELS.get(k, k), f3(r["accuracy"]), f3(r["macro_f1"]),
                     f"{fo['mean']:.3f} ± {fo['std']:.3f}"])
    return table(["Model", "Accuracy (pooled)", "Macro-F1 (pooled)", "Macro-F1 (fold mean ± std)"],
                 rows)


def _per_class(m: dict) -> str:
    rows = []
    for k, r in m["models"].items():
        pc = r["per_class"]
        rows.append([MODEL_LABELS.get(k, k)] + [
            f"{f3(pc[c]['pooled']['f1'])} (P {f3(pc[c]['pooled']['precision'])} / "
            f"R {f3(pc[c]['pooled']['recall'])})" for c in LABELS])
    sup = m["models"]["rf"]["per_class"]
    head = ["Model"] + [f"{c} (n={sup[c]['pooled']['support']})" for c in LABELS]
    return table(head, rows)


def _events(m: dict) -> str:
    def prec(e: dict) -> str:
        return f3(e["event_precision"]) if e.get("event_precision") is not None else "–"

    rows = []
    items = [(MODEL_LABELS.get(k, k), r["events"]) for k, r in m["models"].items()]
    items += [(f"_reference: {k}_", e) for k, e in
              (m.get("event_reference_predictors") or {}).items()]
    for name, e in items:
        rows.append([name] + [x for c in POINT for x in (
            e[c]["detected_of"], e[c].get("n_predicted", "–"), prec(e[c]),
            e[c]["false_alarms"])])
    return table(["Model", "Pothole detected", "Pothole predicted", "Pothole precision",
                  "Pothole false alarms", "Irregularity detected", "Irregularity predicted",
                  "Irregularity precision", "Irregularity false alarms"], rows)


def _confusion(m: dict, key: str = "rf") -> str:
    c = m["models"][key]["confusion"]
    rows = [[f"**{lab}**"] + row for lab, row in zip(c["labels"], c["counts"])]
    return table(["true \\ predicted"] + c["labels"], rows)


def _ablations(m: dict) -> str:
    rows = []
    for a in m.get("ablations") or []:
        if a["macro_f1"] is None:
            continue
        rows.append([a["ablation"], a["n_windows"], f3(a["macro_f1"]),
                     f"{a['macro_f1_fold_mean']:.3f} ± {a['macro_f1_fold_std']:.3f}",
                     a["pothole_events"] or a["impact_events"] or "–",
                     a["irregularity_events"] or "–"])
    return table(["RF variant", "windows", "macro-F1 (pooled)", "fold mean ± std",
                  "Pothole (or Impact) events", "Irregularity events"], rows)


def _variants(art: dict) -> str:
    rows = []
    for name, key, what in (("main (real only)", "metrics", "none"),
                            ("augmentation", "aug", "train folds augmented"),
                            ("calibrated synthetic", "calib", "calibrated synthetic windows "
                                                              "added to train folds")):
        m = art.get(key)
        if not m:
            continue
        r = m["models"]["rf"]
        rows.append([name, what, m["data_provenance"], m["test_data_provenance"],
                     f3(r["macro_f1"]), r["events"]["Pothole"]["detected_of"],
                     r["events"]["Irregularity"]["detected_of"]])
    return table(["RF run", "extra training data", "provenance", "test folds",
                  "macro-F1 (pooled)", "Pothole events", "Irregularity events"], rows)


def _three_class(m: dict) -> str:
    t = m["three_class_diagnostic"]
    rows = [[c, f3(v["precision"]), f3(v["recall"]), f3(v["f1"]), v["support"]]
            for c, v in t["per_class"].items()]
    return table(["class", "precision", "recall", "F1", "windows"], rows)


def _pocket(m: dict) -> str:
    rows = []
    for fs, p in m["pocket_test"].items():
        for c, v in p["per_class"].items():
            pred = ", ".join(f"{k} {n}" for k, n in v["predicted_as"].items() if n)
            rows.append([fs, c, f"{v['windows_correct']}/{v['n_windows']}", pred,
                         p["events_detected"].get(c, "–")])
    return table(["feature set", "class", "windows correct", "predicted as", "events"], rows)


def _speed(m: dict) -> str:
    rows = [[s["label"], s["n"], f"{s['mean_mps']:.2f}", f"{s['std_mps']:.2f}"]
            for s in m.get("speed_by_class") or []]
    return table(["class", "windows", "mean speed (m/s)", "std"], rows)


def _ablation(m: dict, prefix: str) -> dict | None:
    return next((a for a in m.get("ablations") or [] if a["ablation"].startswith(prefix)), None)


def _speed_delta(m: dict) -> float:
    w = _ablation(m, "(i) WITH speed")
    return (w["macro_f1"] - m["models"]["rf"]["macro_f1"]) if w else float("nan")


def _dataset(art: dict) -> str:
    d = art.get("dataset")
    if not d:
        return "_reports/dataset.json missing (run `report --data-dir data/raw` locally)._"
    rows = [[r["ride_id"], r["mount"], r["duration_s"], r["fs_hz"], r["n_gps_fixes"],
             ", ".join(f"{k} {v}" for k, v in r["labelled_intervals"].items()), r["notes"]]
            for r in d["rides"]]
    return table(["ride", "mount", "duration (s)", "sampling (Hz)", "GPS fixes",
                  "labelled intervals", "notes"], rows)


def caveats(art: dict) -> list[str]:
    """Limitations derived from the numbers (each line is triggered by a condition)."""
    m, card, ins = art["metrics"], art["card"], art.get("insample")
    rf = m["models"]["rf"]
    out = []
    n_rides = m["n_rides"]
    km = rf["events"]["distance_km"]
    out.append(f"Tiny dataset: {n_rides} {m.get('mount_filter') or ''} rides, {m['n_windows']} windows, "
               f"{km} km evaluated, one session. Every number here is a property of this "
               f"session, not of cycleways in general.")
    if m["cv_mode"] == "blocked":
        out.append("Blocked CV inside rides (blocks of "
                   f"{m['cv_protocol']['block_s']:.0f} s, purge {m['cv_protocol']['purge_s']:.0f}"
                   " s) is optimistic: test blocks share the road, the day, the phone and the "
                   "rider with the training blocks. Ride-grouped CV is impossible because "
                   + ", ".join(f"{c} lives in {n} ride" for c, n in m["rides_per_class"].items()
                               if n < 2) + ".")
    out.append("Model and training variant were chosen on the same folds that give the reported "
               "estimate (RF vs rule / HGB, real-only vs augmented / calibrated), so the RF "
               "number is slightly optimistic; RF was the default before any real data, and "
               "hyperparameters were never tuned.")
    if ins:
        gap = ins["macro_f1"] - rf["macro_f1"]
        if gap > 0.1:
            out.append(f"Overfitting gap: in-sample macro-F1 {ins['macro_f1']:.3f} vs blocked-CV "
                       f"{rf['macro_f1']:.3f} (gap {gap:.3f}).")
    n_pot = rf["events"]["Pothole"]["n_labelled"]
    if n_pot < 10:
        out.append(f"Pothole has only {n_pot} labelled events (< 10), and they are speed bumps "
                   "used as a pothole proxy: every Pothole number is anecdotal.")
    conf = dict(zip(rf["confusion"]["labels"], rf["confusion"]["counts"]))
    li = rf["confusion"]["labels"]
    pi = conf["Pothole"][li.index("Irregularity")] + conf["Irregularity"][li.index("Pothole")]
    if pi:
        t = m["three_class_diagnostic"]["macro_f1"]
        out.append(f"Pothole / Irregularity confusion: {pi} windows swapped between the two "
                   f"classes; merging them into one Impact class gives macro-F1 {t:.3f} "
                   "(diagnostic only).")
    for n in m.get("experiment_notes") or []:
        if n.startswith("CONFOUNDING"):
            out.append("Speed confound: " + n.split(": ", 1)[1])
    p = m["pocket_test"]["invariant"]["events_detected"]
    out.append("Pocket test inconclusive: only " + ", ".join(f"{c} {v}" for c, v in p.items())
               + f" event(s) on {m['pocket_test']['invariant']['n_windows']} windows.")
    out.append("Severity (1–5) is an uncalibrated heuristic (fixed acceleration bins), not a "
               "measured damage level; segment bands inherit that.")
    if card.get("data_provenance") != "real":
        out.append(f"The model card says data_provenance = {card['data_provenance']}: these are "
                   "NOT real results.")
    return out


def render_results(art: dict) -> str:
    m, card, ins = art["metrics"], art["card"], art.get("insample")
    rf = m["models"]["rf"]
    cp = m["cv_protocol"]
    km = rf["events"]["distance_km"]
    parts = [
        "# Results",
        "",
        "<!-- GENERATED by `python -m smartcycle report` from reports/*.json and "
        "models/model_card.json. Do not edit by hand. -->",
        "",
        f"> **Data provenance: {m['data_provenance'].upper()}** (test folds: "
        f"{m['test_data_provenance']}). Final model: `{card['model']}`, provenance "
        f"**{card['data_provenance']}**, features `{card['feature_set']}`, speed features "
        f"{'on' if card['include_speed'] else 'off'}, {card['window_s']} s windows.",
        f"> {cp['header']}.",
        "",
        "## Dataset",
        "",
        f"{m['provenance_note']}. Mount filter: {m.get('mount_filter')}; "
        f"{m['n_stationary_windows_dropped']} stationary window(s) dropped. Windows per "
        "class: " + ", ".join(f"{c} {n}" for c, n in m["n_windows_per_class"].items())
        + ". Rides per class: " + ", ".join(f"{c} {n}" for c, n in m["rides_per_class"].items())
        + ".",
        "",
        _dataset(art),
        "",
        "## Protocol",
        "",
        f"- Blocked CV: blocks of {cp['block_s']:.0f} s inside each ride ({cp['n_blocks']} "
        f"blocks), {cp['n_splits']} folds (StratifiedGroupKFold over blocks, seed "
        f"{cp['seed']}), training windows within {cp['purge_s']:.0f} s of a test window removed "
        "(purge).",
        "- Why not ride-grouped CV: each of Rough, Pothole and Irregularity was recorded in a "
        "single ride, so holding a ride out removes the class from training.",
        "- Metrics are pooled over all out-of-fold predictions (fold mean ± std in a separate "
        "column). Event metrics: predicted events are cut into units of at most 2 s; each "
        "labelled event is matched one-to-one by a predicted unit of the same type whose peak "
        f"lies within ± {rf['events']['tolerance_s']} s of it; counts only (k/n detected, n "
        f"predicted), every unmatched unit is a false alarm, over {km} km of evaluated riding.",
        "- No tuning on test folds; augmentation / synthetic windows only on training folds.",
        "",
        "## Window-level results",
        "",
        _models_table(m),
        "",
        "Per class F1 (pooled precision / recall):",
        "",
        _per_class(m),
        "",
        "RandomForest confusion matrix (pooled out-of-fold windows):",
        "",
        _confusion(m),
        "",
        "## Event-level results",
        "",
        "Matching (since the P90 review): " + str(rf["events"].get("matching", "")) + ". "
        "Reference rows = degenerate predictors (every evaluated window predicted as one "
        "class) under the same metric; a model is useful only if it beats them on precision "
        "and false alarms, not only on detected events. Before / after the metric change: "
        "docs/EXPERIMENTS.md.",
        "",
        _events(m),
        "",
        "False alarms per km (RF): " + ", ".join(
            f"{c} {rf['events'][c]['false_alarms']} over {km} km "
            f"= {rf['events'][c]['false_alarms_per_km']:.1f} /km" for c in POINT)
        + " (tiny distance: a single false alarm moves this by "
          f"{1 / km:.1f} /km).",
        "",
        "## In-sample vs cross-validated",
        "",
        (f"RF on its own training windows: macro-F1 {ins['macro_f1']:.3f}, accuracy "
         f"{ins['accuracy']:.3f} ({ins['n_windows']} windows) vs blocked-CV macro-F1 "
         f"{rf['macro_f1']:.3f}. The in-sample score is not an evaluation; the gap "
         f"({ins['macro_f1'] - rf['macro_f1']:.3f}) measures overfitting."
         if ins else "_reports/insample.json missing._"),
        "",
        "## Ablations (RF)",
        "",
        _ablations(m),
        "",
        "The 2 s window row is not like-for-like: its windows (same count by coincidence) have "
        "a different class mix (an event inside a 2 s window labels more windows Pothole / "
        "Irregularity), so its macro-F1 is scored on different labels; compare event counts "
        "and false alarms rather than macro-F1.",
        "",
        f"Speed features change the RF macro-F1 by {_speed_delta(m):+.3f}; the rule fixed "
        "before the results was to keep them only above +0.05, so the final model has no "
        "speed features. Speed per class (speed alone already separates classes, a recording "
        "artefact):",
        "",
        _speed(m),
        "",
        "## 3-class diagnostic (Smooth / Rough / Impact = Pothole + Irregularity)",
        "",
        f"Macro-F1 {m['three_class_diagnostic']['macro_f1']:.3f}; Impact events "
        f"{m['three_class_diagnostic']['events']['Impact']['detected_of']}, false alarms "
        f"{m['three_class_diagnostic']['events']['Impact']['false_alarms']}. "
        + m["three_class_diagnostic"]["note"],
        "",
        _three_class(m),
        "",
        "## Augmentation and calibrated synthetic data",
        "",
        _variants(art),
        "",
        "## Pocket test (train handlebar -> test pocket)",
        "",
        _pocket(m),
        "",
        "## Figures",
        "",
        "- `reports/confusion_matrix.png`: confusion matrices (pooled out-of-fold).",
        "- `reports/feature_importance.png`: RF permutation importance.",
        "- `reports/signal_examples.png`: example windows per class (real).",
        "- `reports/synthetic/synth_examples.png`: SYNTHETIC example windows (generator only).",
        "- `docs/screenshots/`: dashboard screenshots.",
        "",
        "## Limitations and honest caveats",
        "",
        *[f"- {c}" for c in caveats(art)],
        "",
    ]
    return "\n".join(parts)
