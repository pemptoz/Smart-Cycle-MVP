"""Build the committed public copy of the real dataset: data/raw -> data/real_public.

Copies the canonical ride CSVs and labels.csv, and writes a rides.csv WITHOUT the `device`
column, with `started_at_utc` from each Phyphox `meta/time.csv` START row. No Phyphox meta
files (device.csv, time.csv) are copied. Rerunnable: `python scripts/make_real_public.py`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import pandas as pd

from smartcycle.io.loader import phyphox_start_ms

PUBLIC_COLUMNS = ["ride_id", "file", "mount", "notes", "started_at_utc"]


def iso_utc(ms: float | None) -> str:
    if ms is None:
        return ""
    d = dt.datetime.fromtimestamp(round(ms) / 1000.0, tz=dt.timezone.utc)
    return d.isoformat(timespec="milliseconds").replace("+00:00", "Z")


MANIFEST_DESC = ("5 real handlebar/pocket rides recorded on 16 Nov 2026 (Phyphox), canonical "
                 "CSV, consented, CC BY 4.0 (DATA_LICENSE.md)")


def build(src: Path, out: Path) -> pd.DataFrame:
    meta = pd.read_csv(src / "rides.csv", dtype=str).fillna("")
    (out / "rides").mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in meta.iterrows():
        rid = r["ride_id"]
        shutil.copyfile(src / r["file"], out / "rides" / f"{rid}.csv")
        rows.append({"ride_id": rid, "file": f"rides/{rid}.csv", "mount": r["mount"],
                     "notes": r["notes"],
                     "started_at_utc": iso_utc(phyphox_start_ms(src / "rides" / rid))})
    shutil.copyfile(src / "labels.csv", out / "labels.csv")
    df = pd.DataFrame(rows, columns=PUBLIC_COLUMNS)
    df.to_csv(out / "rides.csv", index=False)
    # provenance fails closed (smartcycle.io.dataset.read_provenance): state it explicitly
    (out / "MANIFEST.json").write_text(json.dumps({"provenance": "real", "description": MANIFEST_DESC},
                                                  indent=1) + "\n")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", default="data/raw")
    p.add_argument("--out", default="data/real_public")
    a = p.parse_args()
    print(build(Path(a.src), Path(a.out)).to_string(index=False))
