import zipfile

import numpy as np
import pandas as pd
import pytest

from smartcycle.io import CANONICAL_COLUMNS, ColumnError, load_ride, norm_col
from smartcycle.io.loader import to_timestamp_ms


def write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return p


CANONICAL = """timestamp_ms,accel_x,accel_y,accel_z,latitude,longitude,gps_accuracy_m,speed_mps,label
1700468404000,0.1,0.2,9.8,53.3499,-6.2604,5,4.2,Smooth
1700468404020,0.2,0.1,9.7,,,,,Smooth
1700468404040,0.3,0.0,9.9,,,,,Pothole
"""

PLAN = """Timestamp (ms),Accel_X (m/s²),Accel_Y (m/s²),Accel_Z (m/s²),Latitude,Longitude,Ground_Truth_Label
1700468404000,0.1,0.2,9.8,53.3499,-6.2604,Smooth
1700468404020,0.2,0.1,9.7,,,Smooth
1700468404040,0.3,0.0,9.9,,,pothole
"""

PHYPHOX = """"Time (s)","Acceleration with g x (m/s^2)","Acceleration with g y (m/s^2)","Acceleration with g z (m/s^2)","Latitude (°)","Longitude (°)","Horizontal Accuracy (m)","Speed (m/s)"
0.000,0.1,0.2,9.8,,,,
0.010,0.15,0.2,9.8,53.3499,-6.2604,5,4.2
0.020,0.2,0.1,9.7,,,,
0.040,0.3,0.0,9.9,,,,
"""


def test_norm_col():
    assert norm_col("Acceleration with g x (m/s^2)") == ("acceleration with g x", "m/s^2")
    assert norm_col("﻿ Latitude (°) ") == ("latitude", "°")
    assert norm_col("Accel_X [m/s²]") == ("accel x", "m/s²")


def test_three_layouts_give_canonical_schema(tmp_path):
    a = load_ride(write(tmp_path, "a.csv", CANONICAL))
    b = load_ride(write(tmp_path, "b.csv", PLAN))
    for df in (a, b):
        assert list(df.columns) == CANONICAL_COLUMNS + ["t_s"]
        assert df["timestamp_ms"].dtype == np.int64
        assert df["accel_z"].dtype == np.float64
        assert list(df["label"]) == ["Smooth", "Smooth", "Pothole"]
        assert df["t_s"].tolist() == [0.0, 0.02, 0.04]
        assert df.attrs["time_kind"] == "epoch_ms"
    np.testing.assert_allclose(a["accel_x"], b["accel_x"])
    assert b["gps_accuracy_m"].isna().all()  # optional column absent -> NaN


def test_phyphox_gps_on_separate_row_is_merged(tmp_path):
    df = load_ride(write(tmp_path, "p.csv", PHYPHOX))
    assert len(df) == 4  # every row has accel here (the fix at 0.010 s has accel too)
    assert df.attrs["time_kind"] == "relative_s"
    assert df["timestamp_ms"].tolist() == [0, 10, 20, 40]
    assert df["latitude"].notna().sum() == 1
    assert df.loc[1, "speed_mps"] == 4.2


def test_gps_only_rows_attached_to_nearest_accel_row(tmp_path):
    text = """Time (s),Acceleration x (m/s^2),Acceleration y (m/s^2),Acceleration z (m/s^2),Latitude (°),Longitude (°)
0.00,0,0,9.8,,
0.02,0,0,9.8,,
0.021,,,,53.35,-6.26
0.04,0,0,9.8,,
0.06,0,0,9.8,,
"""
    df = load_ride(write(tmp_path, "g.csv", text))
    assert len(df) == 4  # GPS-only row folded in, not kept as an extra row
    assert df["latitude"].tolist()[1] == 53.35
    assert df["latitude"].isna().sum() == 3  # never interpolated


def test_semicolon_decimal_comma_bom(tmp_path):
    text = ("Time (s);Acceleration x (m/s^2);Acceleration y (m/s^2);Acceleration z (m/s^2);"
            "Latitude (°);Longitude (°)\n"
            "0,000;0,1;0,2;9,8;53,3499;-6,2604\n0,020;0,2;0,1;9,7;;\n")
    df = load_ride(write(tmp_path, "s.csv", "﻿" + text))
    assert df["accel_z"].tolist() == [9.8, 9.7]
    assert df["latitude"].iloc[0] == pytest.approx(53.3499)
    assert df["timestamp_ms"].tolist() == [0, 20]


def test_tab_separated(tmp_path):
    text = "time\tax\tay\taz\n0.0\t0\t0\t9.8\n0.01\t0\t0\t9.8\n"
    df = load_ride(write(tmp_path, "t.csv", text))
    assert df["timestamp_ms"].tolist() == [0, 10]


def test_quoted_decimal_comma_in_comma_file(tmp_path):
    text = 'Time (s),Acceleration x (m/s^2),Acceleration y (m/s^2),Acceleration z (m/s^2)\n' \
           '"0,00","0,1","0,2","9,8"\n"0,02","0,1","0,2","9,7"\n'
    df = load_ride(write(tmp_path, "q.csv", text))
    assert df["accel_z"].tolist() == [9.8, 9.7]


@pytest.mark.parametrize("values,hint,kind,expected", [
    ([0.0, 0.01, 0.02], "s", "relative_s", [0, 10, 20]),
    ([0.0, 0.01, 0.02], None, "relative_s", [0, 10, 20]),
    ([0, 10, 20], "ms", "relative_ms", [0, 10, 20]),
    ([0, 10, 20], None, "relative_ms", [0, 10, 20]),
    ([1700468404000, 1700468404010], None, "epoch_ms", [1700468404000, 1700468404010]),
    ([1700468404.0, 1700468404.01], None, "epoch_s", [1700468404000, 1700468404010]),
])
def test_time_unit_detection(values, hint, kind, expected):
    ms, k = to_timestamp_ms(np.array(values, dtype=float), hint)
    assert k == kind
    np.testing.assert_allclose(ms, expected, atol=1e-3)


def test_unsorted_and_duplicates(tmp_path):
    text = "timestamp_ms,accel_x,accel_y,accel_z\n40,3,0,9.8\n0,1,0,9.8\n20,2,0,9.8\n20,9,9,9\n"
    df = load_ride(write(tmp_path, "u.csv", text))
    assert df["timestamp_ms"].tolist() == [0, 20, 40]
    assert df["accel_x"].tolist() == [1, 2, 3]


def test_missing_column_error_lists_found_and_aliases(tmp_path):
    text = "Time (s),Linear Acceleration x (m/s^2),Linear Acceleration y (m/s^2),Foo\n0,0,0,0\n"
    with pytest.raises(ColumnError) as e:
        load_ride(write(tmp_path, "m.csv", text))
    msg = str(e.value)
    assert "accel_x" in msg and "Columns found" in msg and "Foo" in msg
    assert "Aliases tried for accel_z" in msg and "acceleration with g z" in msg
    assert "gravity" in msg  # linear-acceleration hint


def test_invalid_label_rejected(tmp_path):
    text = "timestamp_ms,accel_x,accel_y,accel_z,label\n0,0,0,9.8,Crack\n"
    with pytest.raises(ValueError, match="Unknown label"):
        load_ride(write(tmp_path, "l.csv", text))


def test_phyphox_folder_and_zip(tmp_path):
    acc = ("Time (s),Acceleration x (m/s^2),Acceleration y (m/s^2),Acceleration z (m/s^2),"
           "Absolute acceleration (m/s^2)\n0.0,0,0,9.8,9.8\n0.02,0,0,9.8,9.8\n0.04,0,0,9.8,9.8\n")
    loc = ("Time (s),Latitude (°),Longitude (°),Height (m),Velocity (m/s),Direction (°),"
           "Horizontal Accuracy (m),Vertical Accuracy (m)\n0.019,53.35,-6.26,10,4.5,90,5,8\n")
    d = tmp_path / "exp"
    d.mkdir()
    (d / "Accelerometer.csv").write_text(acc)
    (d / "Location.csv").write_text(loc)
    z = tmp_path / "exp.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("Accelerometer.csv", acc)
        zf.writestr("Location.csv", loc)
    for src in (d, z):
        df = load_ride(src)
        assert len(df) == 3
        assert df.loc[1, "latitude"] == 53.35
        assert df.loc[1, "speed_mps"] == 4.5
        assert df.loc[1, "gps_accuracy_m"] == 5


def test_canonical_roundtrip(tmp_path):
    from smartcycle.io import to_canonical_csv
    df = load_ride(write(tmp_path, "p.csv", PHYPHOX))
    to_canonical_csv(df, tmp_path / "c.csv")
    back = load_ride(tmp_path / "c.csv")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), back.reset_index(drop=True),
                                  check_exact=False, atol=1e-6)
