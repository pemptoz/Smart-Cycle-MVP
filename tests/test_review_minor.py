"""P90 review, minor items: photo pixel limit (SEC-8), priority repairs never list score-0
segments (REQ-8), `report --check` says so on success (REQ-7), warning when the phone samples
below the target rate (ML-8)."""

import io
import logging
from contextlib import closing

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from smartcycle.api import db
from smartcycle.api.photos import PhotoError, clean_photo
from smartcycle.features.preprocess import PipelineConfig, preprocess_ride


def test_photo_over_the_pixel_limit_is_refused_before_decoding():
    im = Image.new("1", (8000, 6000))  # 48 MP but only a few KB: Pillow alone just warns
    buf = io.BytesIO()
    im.save(buf, "PNG")
    assert len(buf.getvalue()) < 100_000
    with pytest.raises(PhotoError, match="too large"):
        clean_photo(buf.getvalue())
    ok = io.BytesIO()
    Image.new("RGB", (640, 480), (10, 20, 30)).save(ok, "PNG")
    assert clean_photo(ok.getvalue())[1] == "png"


def test_priorities_skip_segments_with_nothing_to_repair(tmp_path):
    with closing(db.connect(tmp_path / "p.db")) as c:
        with c:
            for seg, cov in (("s1", 50.0), ("s2", 80.0)):
                c.execute("INSERT INTO coverage (ride_id, segment_id, covered_m, rough_m)"
                          " VALUES ('r1', ?, ?, 0)", (seg, cov))
        assert {s["segment_id"] for s in db.segment_scores(c, None)} == {"s1", "s2"}
        assert db.priorities(c, None) == []  # smooth only: nothing to prioritise
        with c:
            c.execute("INSERT INTO events (id, ride_id, type, segment_id, severity, created_at)"
                      " VALUES ('A-0001', 'r1', 'Pothole', 's2', 4, '2026-11-16')")
        pr = db.priorities(c, None)
    assert [p["segment_id"] for p in pr] == ["s2"] and pr[0]["score"] > 0


def test_report_check_prints_success(capsys, monkeypatch):
    from pathlib import Path

    from smartcycle.cli import main
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    assert main(["report", "--check"]) == 0
    assert "docs up to date" in capsys.readouterr().out


def _ride(fs_native: float, seconds: float = 6.0) -> pd.DataFrame:
    t = np.arange(0, seconds, 1 / fs_native)
    df = pd.DataFrame({"timestamp_ms": (t * 1000).astype("int64"), "accel_x": 0.1,
                       "accel_y": 0.0, "accel_z": 9.81, "latitude": np.nan,
                       "longitude": np.nan, "gps_accuracy_m": np.nan, "speed_mps": np.nan})
    df["t_s"] = t
    df["label"] = pd.Series([pd.NA] * len(t), dtype="string")
    return df


def test_warns_when_native_rate_is_below_target(caplog):
    cfg = PipelineConfig()  # target 50 Hz
    with caplog.at_level(logging.WARNING):
        low = preprocess_ride(_ride(20.0), cfg, "low")
    assert low.native_fs == pytest.approx(20.0, rel=0.01)
    assert any("interpolation artefacts" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        preprocess_ride(_ride(100.0), cfg, "high")
    assert not any("interpolation artefacts" in r.message for r in caplog.records)
