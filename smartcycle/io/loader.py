"""Load any supported ride file into the canonical schema (see CLAUDE.md "Data contract")."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from smartcycle.io.aliases import CANONICAL_COLUMNS, map_columns
from smartcycle.io.csvsniff import decode_bytes, read_csv_any
from smartcycle.io.labels import normalise_labels

log = logging.getLogger(__name__)

FLOAT_COLUMNS = ["accel_x", "accel_y", "accel_z", "latitude", "longitude",
                 "gps_accuracy_m", "speed_mps"]
GPS_COLUMNS = ["latitude", "longitude", "gps_accuracy_m", "speed_mps"]


def to_timestamp_ms(values: np.ndarray, unit_hint: str | None) -> tuple[np.ndarray, str]:
    """Convert a time column to milliseconds. Returns (ms as float array, kind).

    kind is one of "epoch_ms", "epoch_s", "relative_ms", "relative_s".
    Epoch is detected from magnitude (> 1e11 -> ms, 1e9..1e11 -> s); relative times use the
    header unit if any, otherwise the median step (< 1 -> seconds, else ms).
    """
    v = np.asarray(values, dtype="float64")
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise ValueError("Time column is empty")
    med = float(np.median(np.abs(finite)))
    if med > 1e11:
        return v, "epoch_ms"
    if med > 1e9:
        return v * 1000.0, "epoch_s"
    unit = unit_hint
    if unit is None:
        steps = np.diff(np.sort(finite))
        steps = steps[steps > 0]
        unit = "s" if steps.size and float(np.median(steps)) < 1.0 else "ms"
    if unit == "s":
        return v * 1000.0, "relative_s"
    return v, "relative_ms"


def canonicalise(raw: pd.DataFrame, source: str = "<dataframe>", ride_id: str | None = None
                 ) -> pd.DataFrame:
    """Map columns, convert time, merge GPS-only rows onto accel rows, clean and type."""
    cm = map_columns(list(raw.columns))
    df = pd.DataFrame({canon: raw[orig] for canon, orig in cm.mapping.items()})
    for c in CANONICAL_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA if c == "label" else np.nan
    for c in FLOAT_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["label"] = normalise_labels(df["label"])

    ts, kind = to_timestamp_ms(pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(),
                               cm.time_unit_hint)
    df["timestamp_ms"] = ts
    df = df[np.isfinite(df["timestamp_ms"])]

    accel_ok = df[["accel_x", "accel_y", "accel_z"]].notna().all(axis=1)
    acc = df[accel_ok].copy()
    gps_only = df[~accel_ok & df["latitude"].notna() & df["longitude"].notna()]
    if acc.empty:
        raise ValueError(f"{source}: no row with all three accelerometer values")

    acc["timestamp_ms"] = np.round(acc["timestamp_ms"]).astype("int64")
    acc = acc.sort_values("timestamp_ms", kind="stable")
    acc = acc.drop_duplicates("timestamp_ms", keep="first").reset_index(drop=True)

    if not gps_only.empty:
        # GPS fixes recorded on their own rows: attach each fix to the nearest accel sample
        # (never interpolated). Two fixes landing on the same sample: keep the first.
        g = gps_only[["timestamp_ms"] + GPS_COLUMNS].copy()
        g["timestamp_ms"] = np.round(g["timestamp_ms"]).astype("int64")
        g = g.sort_values("timestamp_ms").drop_duplicates("timestamp_ms")
        dt = float(np.median(np.diff(acc["timestamp_ms"]))) if len(acc) > 1 else 0.0
        tol = int(max(100.0, 2.0 * dt))
        keys = acc[["timestamp_ms"]].rename(columns={"timestamp_ms": "acc_t"})
        g = pd.merge_asof(g, keys, left_on="timestamp_ms", right_on="acc_t",
                          direction="nearest", tolerance=tol).dropna(subset=["acc_t"])
        g["acc_t"] = g["acc_t"].astype("int64")
        g = g.drop_duplicates("acc_t", keep="first").set_index("acc_t")
        idx = acc.set_index("timestamp_ms").index
        for c in GPS_COLUMNS:
            fill = g[c].reindex(idx).to_numpy()
            cur = acc[c].to_numpy()
            acc[c] = np.where(np.isnan(cur), fill, cur)

    out = acc[CANONICAL_COLUMNS].copy()
    out["timestamp_ms"] = out["timestamp_ms"].astype("int64")
    for c in FLOAT_COLUMNS:
        out[c] = out[c].astype("float64")
    out["label"] = out["label"].astype("string")
    out["t_s"] = (out["timestamp_ms"] - out["timestamp_ms"].iloc[0]) / 1000.0
    out.attrs.update({"ride_id": ride_id, "source": source, "time_kind": kind,
                      "column_mapping": dict(cm.mapping)})
    return out


def load_ride(path: str | Path, ride_id: str | None = None) -> pd.DataFrame:
    """Load one ride CSV (canonical, plan-style or Phyphox-style) into the canonical schema.

    Columns: timestamp_ms (int64), accel_x/y/z (m/s^2, incl. gravity), latitude, longitude,
    gps_accuracy_m, speed_mps (float64, NaN between GPS fixes), label (string, NA if
    unlabelled), plus helper t_s (seconds from the first sample). Sorted, de-duplicated.
    """
    path = Path(path)
    if path.is_dir() or path.suffix.lower() == ".zip":
        return load_phyphox_dir(path, ride_id=ride_id)
    raw = read_csv_any(path)
    return canonicalise(raw, source=str(path), ride_id=ride_id or path.stem)


def _find(names: list[str], *keys: str) -> str | None:
    for n in names:
        base = Path(n).name.lower()
        if base.endswith(".csv") and any(k in base for k in keys):
            return n
    return None


def load_phyphox_dir(path: str | Path, ride_id: str | None = None) -> pd.DataFrame:
    """Load a Phyphox export split into one CSV per sensor (folder or .zip):
    an accelerometer file (e.g. `Accelerometer.csv`) and an optional `Location.csv`."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = z.namelist()

            def read(n: str) -> pd.DataFrame:
                return read_csv_any(text=decode_bytes(z.read(n)))
            acc_name = _find(names, "accelerometer", "acceleration")
            loc_name = _find(names, "location", "gps")
            if acc_name is None:
                raise ValueError(f"{path}: no accelerometer CSV inside; files: {names}")
            acc = read(acc_name)
            loc = read(loc_name) if loc_name else None
    else:
        names = [str(p) for p in sorted(path.glob("*.csv"))]
        acc_name = _find(names, "accelerometer", "acceleration")
        loc_name = _find(names, "location", "gps")
        if acc_name is None:
            raise ValueError(f"{path}: no accelerometer CSV; files: {names}")
        acc = read_csv_any(acc_name)
        loc = read_csv_any(loc_name) if loc_name else None
    # Stack both files: GPS rows have no accel values and are attached by canonicalise().
    raw = acc if loc is None else pd.concat([acc, loc], ignore_index=True, sort=False)
    out = canonicalise(raw, source=str(path), ride_id=ride_id or path.stem)
    start = phyphox_start_ms(path)
    if start is not None:
        out.attrs["start_epoch_ms"] = start
    return out


def _start_from_table(t: pd.DataFrame) -> float | None:
    cols = {c.strip().lower(): c for c in t.columns}
    ev, et, st = cols.get("event"), cols.get("experiment time"), cols.get("system time")
    if not (ev and et and st):
        return None
    rows = t[t[ev].astype(str).str.strip().str.upper() == "START"]
    if rows.empty:
        return None
    r = rows.iloc[0]
    try:
        v = (float(r[st]) - float(r[et])) * 1000.0
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) and v > 1e11 else None


def phyphox_start_ms(path: str | Path) -> float | None:
    """Epoch ms of experiment time 0 from a Phyphox `meta/time.csv` (first START row:
    system time - experiment time). `path` = export folder, .zip or the time.csv itself.
    None when there is no usable meta file (the ride then keeps its upload time)."""
    path = Path(path)
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                name = next((n for n in z.namelist()
                             if n.replace("\\", "/").lower().endswith("meta/time.csv")), None)
                if name is None:
                    return None
                return _start_from_table(read_csv_any(text=decode_bytes(z.read(name))))
        f = path if path.is_file() else path / "meta" / "time.csv"
        if not f.is_file():
            return None
        return _start_from_table(read_csv_any(f))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        log.warning("cannot read Phyphox start time from %s: %s", path, exc)
        return None


def to_canonical_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write the 9 canonical columns (NaN/NA -> empty cell)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[CANONICAL_COLUMNS].to_csv(path, index=False, na_rep="", float_format="%.6f")


def canonical_text(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df[CANONICAL_COLUMNS].to_csv(buf, index=False, na_rep="", float_format="%.6f")
    return buf.getvalue()
