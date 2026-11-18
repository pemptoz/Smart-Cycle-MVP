"""Read-only summary of severities in a Smart Cycle DB (P06 recalibration check).

    python scripts/severity_stats.py [data/smartcycle_real.db]

Prints, per event type, the peak values (Pothole: peak |a_vert|, Irregularity: peak a_hmag,
Rough: window rms, all m/s²), the severity histogram (UNCALIBRATED HEURISTIC), the segment
band counts and the event dates.
"""

from __future__ import annotations

import collections
import sqlite3
import sys

from smartcycle.api import db
from smartcycle.geo.severity import SEVERITY_BINS


def main(path: str) -> None:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    print(f"DB: {path}")
    for kind, (field, th) in SEVERITY_BINS.items():
        rows = conn.execute("SELECT peak, severity FROM events WHERE type = ? ORDER BY peak",
                            (kind,)).fetchall()
        peaks = ", ".join(f"{r['peak']:.2f}" for r in rows)
        hist = dict(sorted(collections.Counter(r["severity"] for r in rows).items()))
        print(f"{kind:12s} n={len(rows):2d} {field} [{peaks}]  bins={th}  severity hist={hist}")
    bands = collections.Counter(s["band_label"] for s in db.segment_scores(conn, None))
    print("Segment bands:", {b: bands.get(b, 0) for b in ("Good", "Needs attention",
                                                         "Repair soon")})
    dates = conn.execute("SELECT MIN(COALESCE(observed_at, created_at)), "
                         "MAX(COALESCE(observed_at, created_at)), "
                         "SUM(observed_at IS NULL) FROM events").fetchone()
    print(f"Event dates: {dates[0]} .. {dates[1]} ({dates[2]} without a recorded time)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/smartcycle_real.db")
