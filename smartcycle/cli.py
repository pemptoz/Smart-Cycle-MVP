"""Command-line interface: `python -m smartcycle <command>` (contract in CLAUDE.md)."""

from __future__ import annotations

import argparse
import sys

NOT_YET = {
    "validate-labels": "P02", "train": "P02", "evaluate": "P02", "predict": "P02",
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
