import json

import pandas as pd
import pytest

from smartcycle.io import load_dataset, load_labels, load_ride, merge_labels


def ride_csv(path, n=100, dt_ms=10):
    rows = "\n".join(f"{i * dt_ms},0,0,9.8" for i in range(n))
    path.write_text("timestamp_ms,accel_x,accel_y,accel_z\n" + rows + "\n")
    return path


LABELS = """ride_id,t_start_s,t_end_s,label
r1,0.0,1.0,Smooth
r1,0.3,0.5,Pothole
r1,0.4,0.7,irregularity
r2,0.0,0.5,Rough
"""


def test_merge_labels_intervals_priority_and_na(tmp_path):
    (tmp_path / "labels.csv").write_text(LABELS)
    labels = load_labels(tmp_path / "labels.csv")
    df = merge_labels(load_ride(ride_csv(tmp_path / "r1.csv", n=120)), labels, "r1")
    lab = df.set_index("timestamp_ms")["label"]
    assert lab[0] == "Smooth" and lab[290] == "Smooth"
    assert lab[300] == "Pothole" and lab[450] == "Pothole"  # Pothole wins over Irregularity
    assert lab[500] == "Irregularity" and lab[690] == "Irregularity"
    assert lab[700] == "Smooth"
    assert lab[1000:].isna().all()  # t_end exclusive, uncovered -> NA


def test_load_labels_rejects_bad_label_and_bad_interval(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("ride_id,t_start_s,t_end_s,label\nr1,0,1,Crack\n")
    with pytest.raises(ValueError, match="Unknown label"):
        load_labels(p)
    p.write_text("ride_id,t_start_s,t_end_s,label\nr1,2,1,Smooth\n")
    with pytest.raises(ValueError, match="t_end_s"):
        load_labels(p)


def make_dataset(d, synthetic):
    (d / "rides").mkdir(parents=True)
    ride_csv(d / "rides" / "r1.csv")
    ride_csv(d / "rides" / "r2.csv")
    (d / "rides.csv").write_text("ride_id,file,mount,device,notes\n"
                                 "r1,rides/r1.csv,handlebar,dev-a,\nr2,r2.csv,pocket,dev-b,x\n")
    (d / "labels.csv").write_text(LABELS)
    if synthetic:
        (d / "MANIFEST.json").write_text(json.dumps({"provenance": "synthetic", "seed": 0}))


# no MANIFEST.json = unknown provenance (fails closed, P90 review ML-2), never "real"
@pytest.mark.parametrize("synthetic,expected", [(True, "synthetic"), (False, "unknown")])
def test_load_dataset(tmp_path, synthetic, expected):
    make_dataset(tmp_path, synthetic)
    ds = load_dataset(tmp_path)
    assert ds.provenance == expected
    assert set(ds.rides) == {"r1", "r2"}
    assert ds.meta.loc["r2", "mount"] == "pocket"
    assert ds.rides["r2"]["label"].iloc[0] == "Rough"
    assert pd.isna(ds.rides["r2"]["label"].iloc[-1])


def test_load_dataset_missing_rides_csv(tmp_path):
    with pytest.raises(FileNotFoundError, match="rides.csv"):
        load_dataset(tmp_path)
