"""Locale-tolerant CSV reading: BOM, separator sniffing (`,` `;` tab) and decimal comma."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

_DEC_COMMA = re.compile(r"\d,\d")


def decode_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def sniff_format(text: str) -> tuple[str, str]:
    """Return (separator, decimal) for a CSV text."""
    lines = [ln for ln in text.splitlines()[:30] if ln.strip()]
    if not lines:
        raise ValueError("Empty CSV file")
    # The header has no decimal commas, so it is the most reliable place to count separators
    # (csv.Sniffer can be fooled by "1,23;4,56").
    header = re.sub(r'"[^"]*"', "", lines[0])
    counts = {c: header.count(c) for c in (";", "\t", ",")}
    sep = max(counts, key=counts.get) if max(counts.values()) > 0 else None
    if sep is None:
        try:
            sep = csv.Sniffer().sniff("\n".join(lines[:20]), delimiters=",;\t").delimiter
        except csv.Error:
            sep = ","
    decimal = "."
    if sep != ",":
        body = "\n".join(lines[1:])
        if _DEC_COMMA.search(body):
            decimal = ","
    return sep, decimal


def read_csv_any(path: str | Path | None = None, text: str | None = None) -> pd.DataFrame:
    """Read a CSV whatever its locale. Values that remain text with a decimal comma
    (e.g. quoted "1,23" in a comma-separated file) are converted to numbers."""
    if text is None:
        text = decode_bytes(Path(path).read_bytes())
    sep, decimal = sniff_format(text)
    df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, skipinitialspace=True)
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype("string").str.strip()
            conv = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
            # convert only if every non-empty value is numeric (keeps label columns as text)
            nonempty = s.notna() & (s != "")
            if nonempty.any() and conv[nonempty].notna().all():
                df[c] = conv.astype("float64")
    return df
