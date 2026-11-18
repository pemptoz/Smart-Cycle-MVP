"""Ride start time: Phyphox meta/time.csv START -> recorded event dates; otherwise the upload
time, flagged `date_kind = "uploaded"`. Synthetic fixtures only."""

import datetime as dt
import shutil
import sqlite3
import zipfile

import pandas as pd

from smartcycle.cli import main
from smartcycle.io.loader import load_ride, phyphox_start_ms

TIME_CSV = ('"event","experiment time","system time","system time text"\n'
            '"START",0.000000000000000E0,1.794850438036630E9,"2026-11-16 17:33:58.037 UTCZ"\n'
            '"PAUSE",5.781778154167114E1,1.794850495853267E9,"2026-11-16 17:34:55.853 UTCZ"\n')
START_MS = 1794850438036.63


def _phyphox_dir(d):
    acc = ("Time (s),Acceleration x (m/s^2),Acceleration y (m/s^2),Acceleration z (m/s^2)\n"
           "0.0,0,0,9.8\n0.02,0,0,9.8\n0.04,0,0,9.8\n")
    (d / "meta").mkdir(parents=True)
    (d / "Accelerometer.csv").write_text(acc)
    (d / "meta" / "time.csv").write_text(TIME_CSV)
    return d


def test_phyphox_start_from_folder_zip_and_missing(tmp_path):
    d = _phyphox_dir(tmp_path / "exp")
    assert abs(phyphox_start_ms(d) - START_MS) < 1
    z = tmp_path / "exp.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("Accelerometer.csv", (d / "Accelerometer.csv").read_text())
        zf.writestr("meta/time.csv", TIME_CSV)
    assert abs(phyphox_start_ms(z) - START_MS) < 1
    assert abs(load_ride(d).attrs["start_epoch_ms"] - START_MS) < 1
    assert phyphox_start_ms(tmp_path / "nothing") is None
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "time.csv").write_text("a,b\n1,2\n")
    assert phyphox_start_ms(tmp_path / "bad" / "time.csv") is None


def _events(db_path):
    with sqlite3.connect(db_path) as c:
        ev = c.execute("SELECT observed_at, created_at FROM events").fetchall()
        rides = c.execute("SELECT started_at FROM rides").fetchall()
    return ev, rides


def test_ingest_dir_uses_meta_start_time(tmp_path, tiny_data, tiny_model, routes_file):
    root = tmp_path / "raw"
    (root / "canonical").mkdir(parents=True)
    shutil.copy(tiny_data / "rides" / "syn_00.csv", root / "canonical" / "r1.csv")
    pd.DataFrame([{"ride_id": "r1", "file": "canonical/r1.csv", "mount": "handlebar"}]).to_csv(
        root / "rides.csv", index=False)
    (root / "rides" / "r1" / "meta").mkdir(parents=True)
    (root / "rides" / "r1" / "meta" / "time.csv").write_text(TIME_CSV)
    dbp = tmp_path / "t.db"
    assert main(["ingest", "--dir", str(root), "--provenance", "synthetic", "--db", str(dbp),
                 "--model", str(tiny_model[0]), "--network", str(routes_file)]) == 0
    ev, rides = _events(dbp)
    assert rides[0][0].startswith("2026-11-16T17:33")
    assert ev and all(o is not None and o.startswith("2026-11-16T17:3") for o, _ in ev)


def test_csv_without_meta_keeps_upload_time(tmp_path, tiny_data, tiny_model, routes_file):
    dbp = tmp_path / "t.db"
    assert main(["ingest", str(tiny_data / "rides" / "syn_00.csv"), "--provenance",
                 "synthetic", "--db", str(dbp), "--model", str(tiny_model[0]),
                 "--network", str(routes_file)]) == 0
    ev, rides = _events(dbp)
    assert rides[0][0] is None and ev and all(o is None for o, _ in ev)
    from smartcycle.api import db
    with sqlite3.connect(dbp) as c:
        c.row_factory = sqlite3.Row
        evs = db.list_events(c)
    assert all(e["date_kind"] == "uploaded" and e["date"] == e["created_at"] for e in evs)
    # explicit --started-at gives recorded dates
    dbp2 = tmp_path / "t2.db"
    assert main(["ingest", str(tiny_data / "rides" / "syn_00.csv"), "--provenance",
                 "synthetic", "--db", str(dbp2), "--model", str(tiny_model[0]),
                 "--network", str(routes_file), "--started-at", "2026-11-16T10:00:00"]) == 0
    ev, _ = _events(dbp2)
    assert all(dt.datetime.fromisoformat(o).date() == dt.date(2026, 11, 16) for o, _ in ev)
