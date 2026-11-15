"""A labels.csv exported by web/label.html must be accepted by the Python side.

The fixture is produced by the tool's own export code (`toCSV` in the label-core block),
see tests/js/label_core.test.mjs (run `UPDATE_FIXTURE=1 node --test tests/js/label_core.test.mjs`).
"""

from pathlib import Path

import pandas as pd

from smartcycle.io import load_labels, load_ride, merge_labels
from smartcycle.io.labels import LABELS
from smartcycle.models.labels_check import interval_issues

FIXTURE = Path(__file__).parent / "data" / "label_tool_export.csv"


def test_tool_export_parses_with_load_labels():
    df = load_labels(FIXTURE)
    assert list(df.columns) == ["ride_id", "t_start_s", "t_end_s", "label"]
    assert set(df["label"]) <= set(LABELS)
    assert (df["t_end_s"] > df["t_start_s"]).all()
    assert (df["ride_id"] == "tool_ride").all()
    # non-overlapping, contiguous cover of the 60 s ride
    assert (df["t_start_s"].iloc[1:].to_numpy() >= df["t_end_s"].iloc[:-1].to_numpy() - 1e-9).all()
    assert df["t_start_s"].iloc[0] == 0.0 and df["t_end_s"].iloc[-1] == 60.0
    assert interval_issues(df, {"tool_ride": 60.0}) == []


def test_tool_export_merges_into_a_ride(tmp_path):
    n, dt_ms = 6001, 10  # 60 s at 100 Hz, ride-relative ms
    rows = "\n".join(f"{i * dt_ms},0,0,9.8" for i in range(n))
    p = tmp_path / "tool_ride.csv"
    p.write_text("timestamp_ms,accel_x,accel_y,accel_z\n" + rows + "\n")
    ride = merge_labels(load_ride(p), load_labels(FIXTURE), "tool_ride")
    lab = ride.set_index("t_s")["label"]
    assert lab[0.0] == "Smooth"
    assert lab[8.3] == "Pothole"
    assert lab[25.0] == "Rough"
    assert lab[27.5] == "Irregularity"
    assert lab[34.0] == "Smooth"  # the overwritten end of the Rough stretch
    assert lab[41.0] == "Pothole"
    assert pd.isna(lab[60.0])  # t_end is exclusive
