"""Load a whole data directory: rides.csv + ride files + optional labels.csv + provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from smartcycle.io.csvsniff import read_csv_any
from smartcycle.io.labels import load_labels, merge_labels
from smartcycle.io.loader import load_ride

RIDES_COLUMNS = ["ride_id", "file", "mount", "device", "notes"]


@dataclass
class Dataset:
    rides: dict[str, pd.DataFrame]
    meta: pd.DataFrame  # one row per ride, index = ride_id
    provenance: str  # "real" | "synthetic" | "unknown" (see read_provenance)
    labels: pd.DataFrame | None = None
    root: Path = field(default_factory=Path)


PROVENANCES = ("real", "synthetic", "unknown")


def _rides_csv_says_synthetic(root: Path) -> bool:
    """True when rides.csv tags any ride as synthetic (device or notes column)."""
    p = root / "rides.csv"
    if not p.exists():
        return False
    try:
        meta = read_csv_any(p)
    except (ValueError, pd.errors.ParserError):
        return False
    meta.columns = [str(c).strip().lower() for c in meta.columns]
    for c in ("device", "notes"):
        if c in meta and meta[c].astype("string").str.contains("synthetic", case=False,
                                                                na=False).any():
            return True
    return False


def read_provenance(data_dir: str | Path) -> str:
    """`real` | `synthetic` | `unknown`. FAILS CLOSED: only a MANIFEST.json saying
    `"provenance": "real"` makes a directory real. `synthetic*` in the manifest, or a ride tagged
    synthetic in rides.csv (device / notes), gives `synthetic`; anything else is `unknown`
    (train / evaluate then refuse to run without an explicit --provenance)."""
    root = Path(data_dir)
    manifest = root / "MANIFEST.json"
    declared = None
    if manifest.exists():
        try:
            declared = str(json.loads(manifest.read_text()).get("provenance", "")).lower()
        except (json.JSONDecodeError, AttributeError):
            declared = None
    if (declared or "").startswith("synthetic") or _rides_csv_says_synthetic(root):
        return "synthetic"
    return "real" if declared == "real" else "unknown"


def _resolve(root: Path, file: str) -> Path:
    for cand in (root / file, root / "rides" / file):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Ride file {file!r} not found in {root} or {root / 'rides'}")


def load_dataset(data_dir: str | Path, provenance: str | None = None) -> Dataset:
    """`provenance` overrides MANIFEST.json (CLI --provenance); synthetic data stays synthetic."""
    root = Path(data_dir)
    rides_csv = root / "rides.csv"
    if not rides_csv.exists():
        raise FileNotFoundError(f"{rides_csv} not found (expected columns {RIDES_COLUMNS})")
    meta = read_csv_any(rides_csv)
    meta.columns = [c.strip().lower() for c in meta.columns]
    for c in ("ride_id", "file"):
        if c not in meta.columns:
            raise ValueError(f"{rides_csv}: missing column {c!r}; found {list(meta.columns)}")
    for c in RIDES_COLUMNS:
        if c not in meta.columns:
            meta[c] = ""
    meta["ride_id"] = meta["ride_id"].astype("string").str.strip()
    if meta["ride_id"].duplicated().any():
        raise ValueError(f"{rides_csv}: duplicated ride_id")
    labels_csv = root / "labels.csv"
    labels = load_labels(labels_csv) if labels_csv.exists() else None
    rides: dict[str, pd.DataFrame] = {}
    for _, row in meta.iterrows():
        rid = str(row["ride_id"])
        df = load_ride(_resolve(root, str(row["file"])), ride_id=rid)
        if labels is not None:
            df = merge_labels(df, labels, rid)
        rides[rid] = df
    return Dataset(rides=rides, meta=meta.set_index("ride_id"),
                   provenance=resolve_provenance(root, provenance), labels=labels, root=root)


def resolve_provenance(data_dir: str | Path, override: str | None = None) -> str:
    """MANIFEST / rides.csv provenance, or an explicit override. An override can never turn
    data detected as synthetic into real."""
    detected = read_provenance(data_dir)
    if override in (None, "", "auto"):
        return detected
    if override not in PROVENANCES:
        raise ValueError(f"provenance must be one of {PROVENANCES}, got {override!r}")
    if detected == "synthetic" and override != "synthetic":
        raise ValueError(f"{data_dir} is tagged synthetic (MANIFEST.json or rides.csv): "
                         f"--provenance {override} refused")
    return override


def select_mount(ds: Dataset, mount: str = "all") -> Dataset:
    """Keep only the rides of one mount (`all` keeps everything)."""
    if mount == "all":
        return ds
    keep = [r for r in ds.rides if str(ds.meta.loc[r, "mount"]).strip() == mount]
    if not keep:
        raise ValueError(f"no ride with mount={mount!r} in {ds.root}")
    return Dataset(rides={r: ds.rides[r] for r in keep}, meta=ds.meta.loc[keep],
                   provenance=ds.provenance, labels=ds.labels, root=ds.root)
