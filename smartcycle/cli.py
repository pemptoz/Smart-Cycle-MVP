"""Command-line interface: `python -m smartcycle <command>` (contract in CLAUDE.md)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path



def cmd_synth(a: argparse.Namespace) -> int:
    from smartcycle.synth import generate
    from smartcycle.synth.plot import plot_examples

    if a.calibrate_from:
        from smartcycle.synth.calibrate import write_calibrated_dataset
        out = a.out if a.out != "data/synthetic" else "data/synthetic_calibrated"
        m = write_calibrated_dataset(a.calibrate_from, out, n_rides=a.rides or 6, minutes=a.minutes or 1.0, seed=a.seed,
                                     labels=a.labels)
        print(f"Wrote {m['n_rides']} SYNTHETIC-CALIBRATED rides to {out} "
              f"(calibrated from {a.calibrate_from}; windows per kind {m['windows_per_kind']})")
        print(f"Calibration: {json.dumps(m['calibration'])}")
        print("NOTE: training data only, never test folds; swerve/braking pulses are absent "
              "from the real data.")
        return 0
    m = generate(a.out, n_rides=a.rides or 12, seed=a.seed, hard=a.hard, minutes=a.minutes)
    print(f"Wrote {m['n_rides']} SYNTHETIC rides to {a.out} (seed={m['seed']}, hard={m['hard']})")
    if a.plot:
        png = plot_examples(a.out, a.plot)
        print(f"Example windows plot: {png}")
    return 0


def cmd_inspect(a: argparse.Namespace) -> int:
    from smartcycle.io.inspect import inspect_ride

    text, _ = inspect_ride(a.csv, labels_path=a.labels, ride_id=a.ride_id)
    print(text)
    return 0


def cmd_normalize(a: argparse.Namespace) -> int:
    from smartcycle.io.loader import load_ride, to_canonical_csv

    df = load_ride(a.raw, ride_id=a.ride_id)
    to_canonical_csv(df, a.out)
    print(f"{a.raw} -> {a.out}: {len(df)} samples, "
          f"{int(df['latitude'].notna().sum())} GPS fixes, time={df.attrs['time_kind']}")
    print(f"Column mapping: {df.attrs['column_mapping']}")
    return 0


def _cfg(a: argparse.Namespace):
    from smartcycle.features.preprocess import PipelineConfig

    return PipelineConfig(fs=a.fs, window_s=a.window_s, step_s=a.step_s,
                          feature_set=a.feature_set, include_speed=not a.no_speed,
                          max_gps_accuracy_m=(None if a.max_gps_accuracy <= 0
                                              else a.max_gps_accuracy))


def _add_cfg_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--fs", type=float, default=50.0, help="resampling rate (Hz)")
    s.add_argument("--window-s", type=float, default=1.0)
    s.add_argument("--step-s", type=float, default=0.5)
    s.add_argument("--feature-set", default="invariant", choices=["invariant", "invariant+fixed"])
    s.add_argument("--no-speed", action="store_true", help="drop the speed features")
    s.add_argument("--max-gps-accuracy", type=float, default=30.0,
                   help="drop GPS fixes with accuracy above this (m); <= 0 keeps all")


def cmd_validate_labels(a: argparse.Namespace) -> int:
    from smartcycle.models.labels_check import validate_labels

    text, n_warn = validate_labels(a.data_dir, _cfg(a))
    print(text)
    return 0


def _cvc(a: argparse.Namespace):
    from smartcycle.models.evaluation import CVConfig

    return CVConfig(cv=a.cv, block_s=a.block_s, purge_s=a.purge_s,
                    augment=getattr(a, "augment", False),
                    extra_train=getattr(a, "extra_train", None))


def _print_run(run, ablate: bool) -> None:
    from smartcycle.models.evaluation import BLOCKED_HEADER
    from smartcycle.models.train import results_table

    if run.splits.mode == "blocked":
        print(f"\n*** {BLOCKED_HEADER} ***")
    print(f"\nCV {run.splits.mode} ({len(run.splits.folds)} folds), "
          f"pooled out-of-fold metrics:")
    print(results_table(run))
    if run.splits.warning:
        print(f"\n{run.splits.warning}")
    if ablate and run.ablations is not None:
        ab = run.ablations
        print("\nExperiments (RandomForest, same CV protocol):")
        print(ab.drop(columns=["feature_set", "window_s", "speed_features"])
              .to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep="-"))
        if ab.attrs.get("speed_by_class"):
            import pandas as pd
            print("\nSpeed per class (moving windows, m/s):")
            print(pd.DataFrame(ab.attrs["speed_by_class"]).to_string(index=False))
        th = ab.attrs.get("three_class")
        if th:
            print(f"\nDIAGNOSTIC 3 classes (Smooth/Rough/Impact; NOT the product): macro-F1 "
                  f"{th['macro_f1']:.3f}, F1 " + ", ".join(
                      f"{c} {v['f1']:.3f}" for c, v in th["per_class"].items())
                  + f", Impact events {th['events']['Impact']['detected_of']}, false alarms "
                  f"{th['events']['Impact']['false_alarms']}")
        for k, res in (ab.attrs.get("pocket") or {}).items():
            print(f"\n(iv) handlebar -> pocket ({k}) — ONLY {res['classes_in_pocket']} labelled "
                  f"in pocket, no overall score: " + "; ".join(
                      f"{c} windows {v['windows_correct']}/{v['n_windows']} (predicted as "
                      f"{ {p: n for p, n in v['predicted_as'].items() if n} }), events "
                      f"{res['events_detected'].get(c, '-')}"
                      for c, v in res["per_class"].items()))
        for n in ab.attrs.get("notes") or []:
            print(f"\n!!! {n}")
    print("\nArtifacts: " + ", ".join(str(p) for p in run.artifacts))


def _output_dirs(a: argparse.Namespace) -> tuple[str, str]:
    """(models dir, reports dir). Defaults depend on the data provenance so that a SYNTHETIC run
    never overwrites the committed real artifacts: synthetic -> models/synthetic,
    reports/synthetic; real (or real + augmented/synthetic training = mixed) -> models, reports.
    Explicit --out / --reports always win."""
    from smartcycle.io.dataset import read_provenance

    sub = "synthetic" if read_provenance(a.data_dir) == "synthetic" else ""
    out = getattr(a, "out", None) or str(Path("models") / sub)
    reports = a.reports or str(Path("reports") / sub)
    if sub and not (getattr(a, "out", None) and a.reports):
        print(f"Synthetic data: artifacts go to {out}/ and {reports}/ (the real ones in "
              f"models/ and reports/ are left untouched)")
    return out, reports


def cmd_evaluate(a: argparse.Namespace) -> int:
    from smartcycle.models.train import evaluate

    _, reports = _output_dirs(a)
    run = evaluate(a.data_dir, _cfg(a), reports, ablate=a.ablate, cvc=_cvc(a), mount=a.mount)
    _print_run(run, a.ablate)
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from smartcycle.models.train import train

    out, reports = _output_dirs(a)
    run, name, mpath, cpath = train(a.data_dir, _cfg(a), out, reports, a.final_model,
                                    ablate=a.ablate, cvc=_cvc(a), mount=a.mount)
    _print_run(run, a.ablate)
    import json
    card = json.loads(cpath.read_text())
    extra = (f" + {card['n_augmented_windows']} augmented + {card['n_synthetic_windows']} "
             f"synthetic" if card["n_augmented_windows"] or card["n_synthetic_windows"] else "")
    print(f"\nFinal model: {name} fitted on all {run.ds_sum['n_windows']} windows{extra} "
          f"-> {mpath}, {cpath} (data_provenance={run.ds_sum['data_provenance']}, "
          f"git_dirty={card['git_dirty']})")
    return 0


def cmd_predict(a: argparse.Namespace) -> int:
    import pandas as pd

    from smartcycle.io.loader import load_ride
    from smartcycle.models.card import load_model
    from smartcycle.models.predict import predict_ride

    bundle = load_model(a.model)
    df = load_ride(a.csv, ride_id=a.ride_id)
    win, ev = predict_ride(df, bundle)
    prov = bundle.get("data_provenance", "unknown")
    print(f"Model: {bundle['model_name']} (trained on {prov.upper()} data) | ride {a.csv}: "
          f"{len(win)} windows")
    print("Predicted windows per class: " + ", ".join(
        f"{k} {v}" for k, v in win["pred"].value_counts().items()))
    pcols = [c for c in win.columns if c.startswith("p_")]
    show = win[["t_start", "t_end", "speed", "pred", "confidence", *pcols]]
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}") if a.all
              else show.head(a.head).to_string(index=False, float_format=lambda x: f"{x:.2f}")
              + (f"\n... ({len(show) - a.head} more, use --all or --out)"
                 if len(show) > a.head else ""))
        print(f"\nEvents ({len(ev)}):")
        fmt = {c: (lambda x: f"{x:.6f}") for c in ("lat", "lon")}
        print(ev.to_string(index=False, formatters=fmt, float_format=lambda x: f"{x:.2f}")
              if len(ev) else "(none)")
    if a.out:
        win.to_csv(a.out, index=False)
        print(f"Window predictions written to {a.out}")
    return 0


def cmd_fetch_osm(a: argparse.Namespace) -> int:
    from smartcycle.geo.osm import OverpassError, fetch_osm

    try:
        path, n, downloaded = fetch_osm(a.tracks, a.out, a.buffer_m, a.force)
    except OverpassError as e:
        print(f"error: {e}\nThe pipeline still runs without a network file: rides are stored "
              f"with matched=false (raw GPS kept). Retry later or use --out on a cached file.",
              file=sys.stderr)
        return 3
    if downloaded:
        print(f"Downloaded OSM network -> {path}: {n} segments (data (c) OpenStreetMap "
              f"contributors, ODbL)")
    else:
        print(f"Using cached {path} ({n} segments); pass --force to download again")
    return 0


def _ingest_files(a: argparse.Namespace) -> list[tuple[str, str, str, float | None]]:
    """(file, mount, provenance, recording start in epoch ms or None) to ingest.

    With --dir + rides.csv, a canonical ride whose raw Phyphox export is kept in
    `<dir>/rides/<ride_id>/meta/time.csv` gets its START time from there."""

    from smartcycle.io.csvsniff import read_csv_any
    from smartcycle.io.dataset import read_provenance
    from smartcycle.io.loader import phyphox_start_ms

    start_arg = None
    if getattr(a, "started_at", None):
        d = dt.datetime.fromisoformat(a.started_at)
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        start_arg = d.timestamp() * 1000.0

    out = []
    if a.dir:
        root = Path(a.dir)
        prov = read_provenance(root) if a.provenance == "auto" else a.provenance
        rides_csv = root / "rides.csv"
        if rides_csv.exists():
            meta = read_csv_any(rides_csv)
            meta.columns = [c.strip().lower() for c in meta.columns]
            for _, r in meta.iterrows():
                f = root / str(r["file"])
                mount = str(r.get("mount") or a.mount)
                rid = str(r.get("ride_id") or "")
                start = phyphox_start_ms(root / "rides" / rid) if rid else None
                out.append((str(f), mount if mount in ("handlebar", "pocket") else a.mount, prov,
                            start if start is not None else start_arg))
        else:
            out += [(str(f), a.mount, prov, start_arg) for f in sorted(root.rglob("*.csv"))
                    if f.name not in ("labels.csv", "rides.csv")]
    for f in a.csv:
        p = Path(f)
        prov = a.provenance
        if prov == "auto":
            prov = "synthetic" if any(read_provenance(d) == "synthetic"
                                      for d in (p.parent, p.parent.parent)) else "unknown"
        out.append((f, a.mount, prov, start_arg))
    return out


def cmd_ingest(a: argparse.Namespace) -> int:
    from contextlib import closing

    from smartcycle.api import db
    from smartcycle.geo.network import default_network_path, load_network
    from smartcycle.io.loader import load_ride
    from smartcycle.models.card import load_model
    from smartcycle.pipeline import process_ride

    files = _ingest_files(a)
    if not files:
        print("error: nothing to ingest (give CSV files or --dir)", file=sys.stderr)
        return 1
    bundle = load_model(a.model)
    net_path = Path(a.network) if a.network else default_network_path()
    net = load_network(net_path)
    print(f"Model: {bundle['model_name']} (trained on {bundle.get('data_provenance', '?').upper()}"
          f" data) | network: " + (f"{net_path} ({len(net)} segments)" if net else
                                    f"none ({net_path} missing -> matched=false)"))
    dbp = Path(a.db) if a.db else db.default_db_path()
    with closing(db.connect(dbp)) as conn:
        for f, mount, prov, start in files:
            df = load_ride(f)
            if start is not None and "start_epoch_ms" not in df.attrs:
                df.attrs["start_epoch_ms"] = start
            res = process_ride(df, bundle, net, mount=mount)
            rid, evs = db.store_ride(conn, res, device_id=a.device_id or db.new_device_id(),
                                     source=f"cli:{Path(f).name}", provenance=prov, mount=mount)
            g = res.summary["gps"]
            kinds = {}
            for e in evs:
                kinds[e["type"]] = kinds.get(e["type"], 0) + 1
            print(f"{f} -> ride {rid} [{prov}]: {res.summary['duration_s']:.0f} s, "
                  f"{res.summary['distance_m']:.0f} m, matched {g.get('pct_matched', 0)}% "
                  f"(snap median {g.get('snap_median_m')} m), events {kinds or '{}'}, "
                  f"merged into existing {sum(e['merged'] for e in evs)}")
    print(f"Database: {dbp}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from smartcycle.geo.network import default_network_path
    from smartcycle.report.artifacts import compute_example, compute_insample, dataset_info, \
        load_artifacts
    from smartcycle.report.render import render_all

    art = load_artifacts(".")
    if art["metrics"] is None or art["card"] is None:
        print("error: reports/metrics.json and models/model_card.json are needed",
              file=sys.stderr)
        return 1
    if a.data_dir:
        ins = compute_insample(a.data_dir, a.model, art["card"])
        ds = dataset_info(a.data_dir)
        Path("reports/insample.json").write_text(json.dumps(ins, indent=1) + "\n")
        Path("reports/dataset.json").write_text(json.dumps(ds, indent=1) + "\n")
        print(f"reports/insample.json: in-sample macro-F1 {ins['macro_f1']:.3f} on "
              f"{ins['n_windows']} windows (card n_windows match: "
              f"{ins['matches_card_n_windows']})")
        print(f"reports/dataset.json: {ds['n_rides']} rides, {ds['total_duration_s']} s")
        ex = compute_example(a.data_dir, a.model, default_network_path())
        Path("reports/walkthrough_example.json").write_text(json.dumps(ex, indent=1) + "\n")
        print("reports/walkthrough_example.json: worked example "
              f"({ex['ride_id']}, {ex['event']['type']} at {ex['event']['t_start_s']} s)")
        art = load_artifacts(".")
    stale = []
    for path, text in render_all(art, ".").items():
        p = Path(path)
        if a.check:
            if not p.exists() or p.read_text(encoding="utf-8") != text:
                stale.append(path)
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    if stale:
        print("stale (run `python -m smartcycle report`): " + ", ".join(stale), file=sys.stderr)
        return 1
    return 0


REAL_DB = Path("data/smartcycle_real.db")


def cmd_serve(a: argparse.Namespace) -> int:
    import uvicorn

    if a.db:
        os.environ["SMARTCYCLE_DB"] = a.db
    elif not os.environ.get("SMARTCYCLE_DB") and REAL_DB.exists():
        os.environ["SMARTCYCLE_DB"] = str(REAL_DB)  # real-data DB by default when it exists
    print(f"Database: {os.environ.get('SMARTCYCLE_DB') or 'data/smartcycle.db'}")
    print(f"Smart Cycle API on http://{a.host}:{a.port}  (docs: /docs, dashboard: /)")
    uvicorn.run("smartcycle.api.app:get_app", factory=True, host=a.host, port=a.port,
                log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m smartcycle",
                                 description="Smart Cycle MVP: road-quality from phone sensors")
    sub = ap.add_subparsers(dest="command", required=True, metavar="<command>")

    s = sub.add_parser("synth", help="generate a SYNTHETIC dataset")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--rides", type=int, default=None, help="default 12 (6 with --calibrate-from)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--hard", action="store_true", help="more class overlap")
    s.add_argument("--calibrate-from", default=None,
                   help="real labelled data dir (or its canonical/ folder): write rides "
                        "calibrated on it (default out: data/synthetic_calibrated)")
    s.add_argument("--labels", default=None, help="labels.csv for --calibrate-from")
    s.add_argument("--minutes", type=float, default=None, help="fixed ride length (default 3-8)")
    s.add_argument("--plot", default="reports/synthetic/synth_examples.png",
                   help="example-window plot path ('' to skip)")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("inspect", help="report sampling, GPS, gravity and labels of a ride")
    s.add_argument("csv")
    s.add_argument("--labels", default=None, help="labels.csv (default: next to the file)")
    s.add_argument("--ride-id", default=None, help="ride id in labels.csv (default: file stem)")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("normalize", help="convert any supported ride file to canonical CSV")
    s.add_argument("raw", help="CSV file, Phyphox folder or .zip")
    s.add_argument("--out", required=True)
    s.add_argument("--ride-id", default=None)
    s.set_defaults(func=cmd_normalize)

    s = sub.add_parser("validate-labels", help="check the labels of a dataset")
    s.add_argument("--data-dir", required=True)
    _add_cfg_args(s)
    s.set_defaults(func=cmd_validate_labels)

    for name, fn, hlp in (("train", cmd_train, "grouped CV + fit and save the final model"),
                          ("evaluate", cmd_evaluate, "ride-grouped CV evaluation + reports")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--data-dir", required=True)
        s.add_argument("--reports", default=None,
                       help="artifacts directory (default: reports/, or reports/synthetic/ "
                            "for synthetic data)")
        s.add_argument("--ablate", action="store_true", help="run the RF ablations")
        s.add_argument("--cv", default="grouped", choices=["grouped", "blocked", "logo"],
                       help="grouped = ride-grouped k-fold, logo = leave-one-ride-out, blocked = "
                            "time blocks inside rides + purge (tiny single-session data)")
        s.add_argument("--block-s", type=float, default=15.0, help="block length (blocked CV)")
        s.add_argument("--purge-s", type=float, default=2.0,
                       help="train windows this close to a test window are removed (blocked CV)")
        s.add_argument("--mount", default="all", choices=["all", "handlebar", "pocket"],
                       help="rides used for CV / training (pocket stays for the orientation test)")
        s.add_argument("--augment", action="store_true",
                       help="augment TRAINING folds only (amplitude, noise, rotation, time-warp, "
                            "event paste); provenance becomes mixed")
        s.add_argument("--extra-train", default=None, choices=["calibrated"],
                       help="add calibrated synthetic windows to TRAINING folds only (mixed)")
        if name == "train":
            s.add_argument("--out", default=None,
                           help="model directory (default: models/, or models/synthetic/ for "
                                "synthetic data)")
            s.add_argument("--final-model", default="rf", choices=["rf", "hgb", "rule", "best"])
        _add_cfg_args(s)
        s.set_defaults(func=fn)

    s = sub.add_parser("predict", help="classify the windows of one ride + merged events")
    s.add_argument("csv")
    s.add_argument("--model", default="models/model.joblib")
    s.add_argument("--ride-id", default=None)
    s.add_argument("--out", default=None, help="write all window predictions to this CSV")
    s.add_argument("--all", action="store_true", help="print every window")
    s.add_argument("--head", type=int, default=15)
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("fetch-osm", help="download the OSM cycleway/road network (Overpass)")
    s.add_argument("--tracks", required=True, help="ride CSV file or directory of rides")
    s.add_argument("--buffer-m", type=float, default=300.0)
    s.add_argument("--out", default="data/osm/cycleways.geojson")
    s.add_argument("--force", action="store_true", help="download even if --out exists")
    s.set_defaults(func=cmd_fetch_osm)

    s = sub.add_parser("ingest", help="run the full pipeline on rides and store them in the DB")
    s.add_argument("csv", nargs="*", help="ride CSV file(s)")
    s.add_argument("--dir", default=None, help="directory of rides (uses rides.csv if present)")
    s.add_argument("--mount", default="handlebar", choices=["handlebar", "pocket"])
    s.add_argument("--provenance", default="auto", choices=["auto", "real", "synthetic",
                                                             "unknown"],
                   help="auto = synthetic if a MANIFEST.json says so, else unknown (--dir: real)")
    s.add_argument("--device-id", default=None)
    s.add_argument("--started-at", default=None,
                   help="ISO recording start for ride-relative timestamps (default: Phyphox "
                        "meta/time.csv when found, else the upload time is shown)")
    s.add_argument("--model", default="models/model.joblib")
    s.add_argument("--network", default=None, help="default: $SMARTCYCLE_NETWORK or "
                                                   "data/osm/cycleways.geojson")
    s.add_argument("--db", default=None, help="default: $SMARTCYCLE_DB or data/smartcycle.db")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("serve", help="run the API + dashboard")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--db", default=None, help="default: $SMARTCYCLE_DB, else "
                   "data/smartcycle_real.db if it exists, else data/smartcycle.db")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("report", help="generate docs/RESULTS.md and the docs from the artifacts")
    s.add_argument("--data-dir", default=None,
                   help="real dataset (local only): also (re)compute reports/insample.json "
                        "(saved model, no retraining) and reports/dataset.json")
    s.add_argument("--model", default="models/model.joblib")
    s.add_argument("--check", action="store_true",
                   help="do not write; exit 1 if a generated doc differs from the file")
    s.set_defaults(func=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
