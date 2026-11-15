"""Command-line interface: `python -m smartcycle <command>` (contract in CLAUDE.md)."""

from __future__ import annotations

import argparse
import sys

NOT_YET = {
    "fetch-osm": "P03", "ingest": "P03", "serve": "P03", "report": "P06",
}


def cmd_synth(a: argparse.Namespace) -> int:
    from smartcycle.synth import generate
    from smartcycle.synth.plot import plot_examples

    m = generate(a.out, n_rides=a.rides, seed=a.seed, hard=a.hard, minutes=a.minutes)
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
                          feature_set=a.feature_set, include_speed=not a.no_speed)


def _add_cfg_args(s: argparse.ArgumentParser) -> None:
    s.add_argument("--fs", type=float, default=50.0, help="resampling rate (Hz)")
    s.add_argument("--window-s", type=float, default=1.0)
    s.add_argument("--step-s", type=float, default=0.5)
    s.add_argument("--feature-set", default="invariant", choices=["invariant", "invariant+fixed"])
    s.add_argument("--no-speed", action="store_true", help="drop the speed features")


def cmd_validate_labels(a: argparse.Namespace) -> int:
    from smartcycle.models.labels_check import validate_labels

    text, n_warn = validate_labels(a.data_dir, _cfg(a))
    print(text)
    return 0


def _print_run(run, ablate: bool) -> None:
    from smartcycle.models.train import results_table

    print(f"\nRide-grouped CV ({run.splits.mode}, {len(run.splits.folds)} folds), "
          f"pooled out-of-fold metrics:")
    print(results_table(run))
    if run.splits.warning:
        print(f"\n{run.splits.warning}")
    if ablate and run.ablations is not None:
        print("\nAblations (RandomForest):")
        print(run.ablations.drop(columns=["feature_set", "window_s", "speed_features"])
              .to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep="-"))
    print("\nArtifacts: " + ", ".join(str(p) for p in run.artifacts))


def cmd_evaluate(a: argparse.Namespace) -> int:
    from smartcycle.models.train import evaluate

    run = evaluate(a.data_dir, _cfg(a), a.reports, ablate=a.ablate)
    _print_run(run, a.ablate)
    return 0


def cmd_train(a: argparse.Namespace) -> int:
    from smartcycle.models.train import train

    run, name, mpath, cpath = train(a.data_dir, _cfg(a), a.out, a.reports, a.final_model,
                                    ablate=a.ablate)
    _print_run(run, a.ablate)
    print(f"\nFinal model: {name} fitted on all {run.ds_sum['n_windows']} windows "
          f"-> {mpath}, {cpath} (data_provenance={run.ds_sum['data_provenance']})")
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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m smartcycle",
                                 description="Smart Cycle MVP: road-quality from phone sensors")
    sub = ap.add_subparsers(dest="command", required=True, metavar="<command>")

    s = sub.add_parser("synth", help="generate a SYNTHETIC dataset")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--rides", type=int, default=12)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--hard", action="store_true", help="more class overlap")
    s.add_argument("--minutes", type=float, default=None, help="fixed ride length (default 3-8)")
    s.add_argument("--plot", default="reports/synth_examples.png",
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
        s.add_argument("--reports", default="reports", help="artifacts directory")
        s.add_argument("--ablate", action="store_true", help="run the RF ablations")
        if name == "train":
            s.add_argument("--out", default="models")
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

    helps = {"validate-labels": "check labels of a dataset", "train": "train models",
             "evaluate": "grouped-CV evaluation", "predict": "classify a ride",
             "fetch-osm": "download the OSM network", "ingest": "ingest a ride into the DB",
             "serve": "run the API + dashboard", "report": "build report assets"}
    for name, phase in NOT_YET.items():
        s = sub.add_parser(name, help=f"{helps[name]} (phase {phase})")
        s.add_argument("args", nargs="*")
        s.add_argument("--data-dir", default=None)
        s.add_argument("--tracks", default=None)
        s.set_defaults(func=lambda a, n=name, ph=phase: _not_yet(n, ph))
    return ap


def _not_yet(name: str, phase: str) -> int:
    print(f"{name}: not implemented yet, phase {phase}")
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
