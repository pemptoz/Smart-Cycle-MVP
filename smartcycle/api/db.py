"""SQLite storage (stdlib sqlite3, parameterised SQL only).

Tables: rides, events, event_reports (which rides back which event, needed for dedup and GDPR
deletion), coverage (distance ridden per segment per ride), ride_track (downsampled matched
track for replay). Anonymous random device ids only, no accounts.

Dedup: a new Pothole / Irregularity within DEDUP_M on the same segment and of the same type as
an existing event adds a report to it (reports = number of distinct rides) and keeps the max
severity instead of creating a new row.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import secrets
import sqlite3
from pathlib import Path

from smartcycle.features.preprocess import haversine_m
from smartcycle.geo.network import Network
from smartcycle.geo.severity import BANDS, band, segment_score
from smartcycle.pipeline import RideResult

DEFAULT_DB = "data/smartcycle.db"
DEDUP_M = 12.0
DEDUP_TYPES = ("Pothole", "Irregularity")
ALERT_MIN_SEVERITY = 4
ALERT_DAYS = 10
ALERT_TOP = 4
PRIORITY_TOP = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS rides (
  id TEXT PRIMARY KEY, device_id TEXT NOT NULL, started_at TEXT, source TEXT, mount TEXT,
  provenance TEXT NOT NULL, n_samples INTEGER, duration_s REAL, distance_m REAL,
  summary TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, ride_id TEXT NOT NULL,
  type TEXT NOT NULL, lat REAL, lon REAL, snapped_lat REAL, snapped_lon REAL, segment_id TEXT,
  way_name TEXT, t_start_ms INTEGER, t_end_ms INTEGER, severity INTEGER, confidence REAL,
  peak REAL, reports INTEGER NOT NULL DEFAULT 1, detail_reason TEXT, detail_text TEXT,
  photo_path TEXT, length_m REAL, observed_at TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS events_seg ON events(segment_id, type);
CREATE TABLE IF NOT EXISTS event_reports (
  event_id TEXT NOT NULL, ride_id TEXT NOT NULL, t_peak_ms INTEGER, t_start_ms INTEGER,
  t_end_ms INTEGER, severity INTEGER, PRIMARY KEY (event_id, ride_id));
CREATE TABLE IF NOT EXISTS coverage (
  ride_id TEXT NOT NULL, segment_id TEXT NOT NULL, covered_m REAL NOT NULL,
  rough_m REAL NOT NULL DEFAULT 0, PRIMARY KEY (ride_id, segment_id));
CREATE TABLE IF NOT EXISTS ride_track (ride_id TEXT PRIMARY KEY, geojson TEXT NOT NULL);
"""


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _parse(ts: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(ts)
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def default_db_path() -> Path:
    return Path(os.environ.get("SMARTCYCLE_DB") or DEFAULT_DB)


def connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def new_device_id() -> str:
    return "anon-" + secrets.token_hex(6)


# ---------------------------------------------------------------- write
def store_ride(conn: sqlite3.Connection, res: RideResult, *, device_id: str, source: str,
               provenance: str, mount: str) -> tuple[str, list[dict]]:
    """Insert a processed ride. Returns (ride_id, events as stored, each with `id` and
    `merged` = True when it was added as a report to an existing event)."""
    ride_id = secrets.token_hex(6)
    created = now_iso()
    s = res.summary
    out: list[dict] = []
    with conn:
        conn.execute(
            "INSERT INTO rides (id, device_id, started_at, source, mount, provenance, n_samples,"
            " duration_s, distance_m, summary, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ride_id, device_id, s.get("started_at"), source, mount, provenance,
             s.get("n_samples"), s.get("duration_s"), s.get("distance_m"), json.dumps(s),
             created))
        for e in res.events:
            eid = _find_duplicate(conn, e)
            merged = eid is not None
            if not merged:
                cur = conn.execute(
                    "INSERT INTO events (ride_id, type, lat, lon, snapped_lat, snapped_lon,"
                    " segment_id, way_name, t_start_ms, t_end_ms, severity, confidence, peak,"
                    " reports, length_m, observed_at, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?)",
                    (ride_id, e["type"], e["lat"], e["lon"], e["snapped_lat"], e["snapped_lon"],
                     e["segment_id"], e["way_name"], e["t_start_ms"], e["t_end_ms"],
                     e["severity"], e["confidence"], e["peak"], e["length_m"],
                     e["observed_at"], created))
                eid = f"A-{cur.lastrowid:04d}"
                conn.execute("UPDATE events SET id = ? WHERE seq = ?", (eid, cur.lastrowid))
            conn.execute(
                "INSERT OR IGNORE INTO event_reports (event_id, ride_id, t_peak_ms, t_start_ms,"
                " t_end_ms, severity) VALUES (?,?,?,?,?,?)",
                (eid, ride_id, e["t_peak_ms"], e["t_start_ms"], e["t_end_ms"], e["severity"]))
            if merged:
                conn.execute(
                    "UPDATE events SET reports = (SELECT COUNT(*) FROM event_reports"
                    " WHERE event_id = ?), severity = MAX(severity, ?),"
                    " peak = MAX(COALESCE(peak, 0), COALESCE(?, 0)) WHERE id = ?",
                    (eid, e["severity"], e["peak"], eid))
            out.append({**e, "id": eid, "merged": merged})
        conn.executemany(
            "INSERT INTO coverage (ride_id, segment_id, covered_m, rough_m) VALUES (?,?,?,?)",
            [(ride_id, c["segment_id"], c["covered_m"], c["rough_m"]) for c in res.coverage])
        conn.execute("INSERT INTO ride_track (ride_id, geojson) VALUES (?,?)",
                     (ride_id, json.dumps(res.track)))
    return ride_id, out


def _find_duplicate(conn: sqlite3.Connection, e: dict) -> str | None:
    if e["type"] not in DEDUP_TYPES or not e.get("segment_id"):
        return None
    rows = conn.execute("SELECT id, snapped_lat, snapped_lon FROM events"
                        " WHERE type = ? AND segment_id = ?", (e["type"], e["segment_id"]))
    best, best_d = None, DEDUP_M
    for r in rows:
        d = float(haversine_m(r["snapped_lat"], r["snapped_lon"], e["snapped_lat"],
                              e["snapped_lon"]))
        if d <= best_d:
            best, best_d = r["id"], d
    return best


def delete_ride(conn: sqlite3.Connection, ride_id: str, photos_dir: str | Path) -> dict | None:
    """GDPR deletion. Events reported only by this ride are deleted (with their photo files);
    merged events lose this ride's report. Returns counts, or None if the ride does not exist."""
    if conn.execute("SELECT 1 FROM rides WHERE id = ?", (ride_id,)).fetchone() is None:
        return None
    deleted, updated, photos = 0, 0, 0
    with conn:
        eids = [r[0] for r in conn.execute(
            "SELECT event_id FROM event_reports WHERE ride_id = ?", (ride_id,))]
        for eid in eids:
            conn.execute("DELETE FROM event_reports WHERE event_id = ? AND ride_id = ?",
                         (eid, ride_id))
            rest = conn.execute("SELECT ride_id, severity FROM event_reports WHERE event_id = ?"
                                " ORDER BY ride_id", (eid,)).fetchall()
            if not rest:
                row = conn.execute("SELECT photo_path FROM events WHERE id = ?",
                                   (eid,)).fetchone()
                if row and row["photo_path"] and _unlink_photo(photos_dir, row["photo_path"]):
                    photos += 1
                conn.execute("DELETE FROM events WHERE id = ?", (eid,))
                deleted += 1
            else:
                conn.execute(
                    "UPDATE events SET reports = ?, severity = ?,"
                    " ride_id = CASE WHEN ride_id = ? THEN ? ELSE ride_id END WHERE id = ?",
                    (len(rest), max(r["severity"] or 1 for r in rest), ride_id,
                     rest[0]["ride_id"], eid))
                updated += 1
        conn.execute("DELETE FROM coverage WHERE ride_id = ?", (ride_id,))
        conn.execute("DELETE FROM ride_track WHERE ride_id = ?", (ride_id,))
        conn.execute("DELETE FROM rides WHERE id = ?", (ride_id,))
    return {"ride_id": ride_id, "events_deleted": deleted, "events_updated": updated,
            "photos_deleted": photos}


def _unlink_photo(photos_dir: str | Path, name: str) -> bool:
    base = Path(photos_dir).resolve()
    p = (base / Path(name).name).resolve()
    if p.parent != base or not p.exists():
        return False
    p.unlink()
    return True


def set_detail(conn: sqlite3.Connection, event_id: str, reason: str, text: str | None) -> bool:
    with conn:
        cur = conn.execute("UPDATE events SET detail_reason = ?, detail_text = ? WHERE id = ?",
                           (reason, text, event_id))
    return cur.rowcount > 0


def set_photo(conn: sqlite3.Connection, event_id: str, name: str) -> str | None:
    """Attach a photo; returns the previous photo name (to delete) if any."""
    with conn:
        row = conn.execute("SELECT photo_path FROM events WHERE id = ?", (event_id,)).fetchone()
        conn.execute("UPDATE events SET photo_path = ? WHERE id = ?", (name, event_id))
    return row["photo_path"] if row else None


# ---------------------------------------------------------------- read
def get_event(conn: sqlite3.Connection, event_id: str) -> dict | None:
    r = conn.execute("SELECT e.*, r.provenance FROM events e LEFT JOIN rides r"
                     " ON r.id = e.ride_id WHERE e.id = ?", (event_id,)).fetchone()
    return _event_row(r) if r else None


def _event_date(d: dict) -> None:
    """`date` = recording time when known (epoch timestamps or Phyphox meta/time.csv), else the
    upload time; `date_kind` says which ("recorded" | "uploaded")."""
    d["date"] = d.get("observed_at") or d["created_at"]
    d["date_kind"] = "recorded" if d.get("observed_at") else "uploaded"


def _event_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    _event_date(d)
    return d


def list_events(conn: sqlite3.Connection, type_: str | None = None,
                bbox: tuple[float, float, float, float] | None = None) -> list[dict]:
    """bbox = (min_lon, min_lat, max_lon, max_lat) on the displayed (snapped) position."""
    sql = ("SELECT e.*, r.provenance FROM events e LEFT JOIN rides r ON r.id = e.ride_id"
           " WHERE 1 = 1")
    args: list = []
    if type_:
        sql += " AND e.type = ?"
        args.append(type_)
    if bbox:
        sql += (" AND COALESCE(e.snapped_lon, e.lon) BETWEEN ? AND ?"
                " AND COALESCE(e.snapped_lat, e.lat) BETWEEN ? AND ?")
        args += [bbox[0], bbox[2], bbox[1], bbox[3]]
    sql += " ORDER BY e.seq"
    return [_event_row(r) for r in conn.execute(sql, args)]


def list_rides(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT id, started_at, source, mount, provenance, n_samples, duration_s,"
        " distance_m, created_at FROM rides ORDER BY created_at DESC")]


def get_ride(conn: sqlite3.Connection, ride_id: str) -> dict | None:
    r = conn.execute("SELECT * FROM rides WHERE id = ?", (ride_id,)).fetchone()
    if r is None:
        return None
    ride = dict(r)
    ride["summary"] = json.loads(ride["summary"] or "{}")
    t = conn.execute("SELECT geojson FROM ride_track WHERE ride_id = ?", (ride_id,)).fetchone()
    ride["track"] = json.loads(t["geojson"]) if t else None
    ride["events"] = [dict(x) for x in conn.execute(
        "SELECT e.*, er.t_peak_ms AS t_peak_ms, er.t_start_ms AS ride_t_start_ms,"
        " er.t_end_ms AS ride_t_end_ms, r.provenance FROM event_reports er"
        " JOIN events e ON e.id = er.event_id LEFT JOIN rides r ON r.id = e.ride_id"
        " WHERE er.ride_id = ? ORDER BY er.t_peak_ms", (ride_id,))]
    for e in ride["events"]:
        _event_date(e)
        # times of THIS ride's report (a merged event keeps the first ride's own times)
        e["t_start_ms"], e["t_end_ms"] = e.pop("ride_t_start_ms"), e.pop("ride_t_end_ms")
    return ride


def provenance_values(conn: sqlite3.Connection) -> list[str]:
    """Distinct ride provenances (sorted): lets the UI tell real+unknown from real+synthetic."""
    return sorted(r[0] for r in conn.execute("SELECT DISTINCT provenance FROM rides"))


def provenance_summary(conn: sqlite3.Connection) -> str:
    """'none' | 'real' | 'synthetic' | 'unknown' | 'mixed' over all stored rides."""
    vals = {r[0] for r in conn.execute("SELECT DISTINCT provenance FROM rides")}
    if not vals:
        return "none"
    return vals.pop() if len(vals) == 1 else "mixed"


def segment_scores(conn: sqlite3.Connection, net: Network | None) -> list[dict]:
    """Per segment with coverage or events: covered_m, rough_m, severity sum, score, band.
    Score / band are an UNCALIBRATED HEURISTIC (see smartcycle.geo.severity)."""
    cov = {r["segment_id"]: (r["c"], r["rm"]) for r in conn.execute(
        "SELECT segment_id, SUM(covered_m) AS c, SUM(rough_m) AS rm FROM coverage"
        " GROUP BY segment_id")}
    ev = {r["segment_id"]: (r["s"] or 0, r["n"]) for r in conn.execute(
        "SELECT segment_id, SUM(CASE WHEN type IN ('Pothole', 'Irregularity') THEN severity"
        " ELSE 0 END) AS s, COUNT(*) AS n FROM events WHERE segment_id IS NOT NULL"
        " GROUP BY segment_id")}
    out = []
    for seg in sorted(set(cov) | set(ev)):
        covered, rough = cov.get(seg, (0.0, 0.0))
        sev, n = ev.get(seg, (0, 0))
        sc = segment_score(sev, rough, covered)
        b = band(sc)
        row = {"segment_id": seg, "covered_m": round(covered, 1), "rough_m": round(rough, 1),
               "severity_sum": int(sev), "n_events": int(n), "score": round(sc, 2), "band": b,
               "band_label": BANDS[b]}
        if net is not None and seg in net.index:
            p = net.props[net.index[seg]]
            row.update({"name": p.get("name"), "highway": p.get("highway"),
                        "is_cycleway": p.get("is_cycleway"), "length_m": p.get("length_m")})
        out.append(row)
    return out


def alerts(conn: sqlite3.Connection, now: dt.datetime | None = None) -> dict:
    """Events with severity >= 4 in the last 10 days, top 4. 'Now' is the wall clock unless no
    event falls in the last 10 days of it (old or future-dated demo data): then the latest event
    date is used as the reference, and the response says so."""
    now = now or dt.datetime.now(dt.timezone.utc)
    evs = list_events(conn)
    dates = [(_parse(e["date"]), e) for e in evs]
    window = dt.timedelta(days=ALERT_DAYS)
    recent = [d for d, _ in dates if now - window <= d <= now]
    ref, reference, note = now, "wall_clock", None
    if not recent and dates:
        ref = max(d for d, _ in dates)
        reference = "latest_event"
        note = (f"No event in the {ALERT_DAYS} days before the current date: the latest event "
                f"date ({ref.isoformat(timespec='seconds')}) is used as 'now'.")
    sel = [e for d, e in dates if ref - window <= d <= ref
           and (e["severity"] or 0) >= ALERT_MIN_SEVERITY]
    sel = sorted(sel, key=lambda e: (e["severity"] or 0, e["date"]), reverse=True)[:ALERT_TOP]
    return {"reference": reference, "reference_time": ref.isoformat(timespec="seconds"),
            "window_days": ALERT_DAYS, "min_severity": ALERT_MIN_SEVERITY, "note": note,
            "alerts": sel}


def priorities(conn: sqlite3.Connection, net: Network | None) -> list[dict]:
    return sorted(segment_scores(conn, net), key=lambda s: -s["score"])[:PRIORITY_TOP]
