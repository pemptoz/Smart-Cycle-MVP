"""Column-name alias table: canonical, plan-style and Phyphox-style headers.

Matching is case-, whitespace-, punctuation- and unit-insensitive: the unit in
parentheses/brackets is split off (and kept as a hint, e.g. for time "s" vs "ms").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CANONICAL_COLUMNS = [
    "timestamp_ms",
    "accel_x",
    "accel_y",
    "accel_z",
    "latitude",
    "longitude",
    "gps_accuracy_m",
    "speed_mps",
    "label",
]
REQUIRED = ["timestamp_ms", "accel_x", "accel_y", "accel_z"]

# Keys are canonical names, values are *normalised* header names (see norm_col).
# "time" stands for the time column whatever its unit; the unit is resolved later.
ALIASES: dict[str, list[str]] = {
    "timestamp_ms": ["timestamp ms", "timestamp", "time ms", "time", "t", "time s", "timestamp s",
                     "elapsed time", "epoch ms", "unix time"],
    "accel_x": ["accel x", "accelx", "acceleration x", "acceleration with g x", "acc x", "ax", "x"],
    "accel_y": ["accel y", "accely", "acceleration y", "acceleration with g y", "acc y", "ay", "y"],
    "accel_z": ["accel z", "accelz", "acceleration z", "acceleration with g z", "acc z", "az", "z"],
    "latitude": ["latitude", "lat", "gps latitude"],
    "longitude": ["longitude", "lon", "lng", "long", "gps longitude"],
    "gps_accuracy_m": ["gps accuracy m", "gps accuracy", "horizontal accuracy", "accuracy",
                       "hacc", "horizontal accuracy m"],
    "speed_mps": ["speed mps", "speed", "gps speed", "velocity"],
    "label": ["label", "ground truth label", "ground truth", "class"],
}

_UNIT_RE = re.compile(r"[\(\[]([^\)\]]*)[\)\]]")


def norm_col(name: str) -> tuple[str, str | None]:
    """Normalise a header. Returns (normalised name, unit or None).

    >>> norm_col("Acceleration with g x (m/s^2)")
    ('acceleration with g x', 'm/s^2')
    >>> norm_col("Timestamp_ms")
    ('timestamp ms', None)
    """
    s = str(name).replace("﻿", "").strip()
    unit = None
    m = _UNIT_RE.search(s)
    if m:
        unit = m.group(1).strip().lower() or None
        s = _UNIT_RE.sub(" ", s)
    s = s.lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s, unit


class ColumnError(ValueError):
    """Raised when a required column cannot be matched to any alias."""


@dataclass
class ColumnMapping:
    mapping: dict[str, str] = field(default_factory=dict)  # canonical -> original header
    time_unit_hint: str | None = None  # "s", "ms" or None


def time_unit_from_header(norm: str, unit: str | None) -> str | None:
    if unit in ("s", "sec", "seconds"):
        return "s"
    if unit in ("ms", "msec", "milliseconds"):
        return "ms"
    if norm.endswith(" ms") or norm == "epoch ms":
        return "ms"
    if norm.endswith(" s"):
        return "s"
    return None


def map_columns(columns: list[str]) -> ColumnMapping:
    """Map raw headers to canonical names. Raises ColumnError if a required one is missing."""
    normed = {c: norm_col(c) for c in columns}
    out = ColumnMapping()
    used: set[str] = set()
    for canon, aliases in ALIASES.items():
        for alias in aliases:  # alias order = priority
            hit = next((c for c in columns if c not in used and normed[c][0] == alias), None)
            if hit is not None:
                out.mapping[canon] = hit
                used.add(hit)
                if canon == "timestamp_ms":
                    out.time_unit_hint = time_unit_from_header(*normed[hit])
                break
    missing = [c for c in REQUIRED if c not in out.mapping]
    if missing:
        lines = [f"Cannot find required column(s): {', '.join(missing)}.",
                 f"Columns found: {list(columns)}",
                 f"Normalised as: {[normed[c][0] for c in columns]}"]
        for c in missing:
            lines.append(f"Aliases tried for {c}: {ALIASES[c]}")
        if any("linear acceleration" in normed[c][0] for c in columns):
            lines.append("Hint: 'Linear Acceleration' excludes gravity; export acceleration "
                         "WITH gravity (Phyphox 'Acceleration with g').")
        raise ColumnError("\n".join(lines))
    return out
