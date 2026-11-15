import json

import numpy as np
import pandas as pd
import pytest

from smartcycle.cli import main
from smartcycle.io import LABELS, load_dataset, load_ride
from smartcycle.synth import generate
from smartcycle.synth.routes import BBOX


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    generate(d, n_rides=4, seed=1, minutes=2.0)
    return d


def test_determinism(tmp_path, synth_dir):
    generate(tmp_path, n_rides=4, seed=1, minutes=2.0)
    for f in ["rides.csv", "labels.csv", "MANIFEST.json", "routes.geojson",
              *[f"rides/syn_0{i}.csv" for i in range(4)]]:
        assert (tmp_path / f).read_bytes() == (synth_dir / f).read_bytes(), f


def test_different_seed_differs(tmp_path, synth_dir):
    generate(tmp_path, n_rides=1, seed=2, minutes=2.0)
    assert (tmp_path / "rides/syn_00.csv").read_bytes() != \
        (synth_dir / "rides/syn_00.csv").read_bytes()


def test_manifest_provenance_and_layouts(synth_dir):
    assert json.loads((synth_dir / "MANIFEST.json").read_text())["provenance"] == "synthetic"
    geo = json.loads((synth_dir / "routes.geojson").read_text())
    assert geo["synthetic"] is True
    ds = load_dataset(synth_dir)
    assert ds.provenance == "synthetic"
    kinds = {df.attrs["time_kind"] for df in ds.rides.values()}
    assert kinds == {"relative_s", "epoch_ms"}
    assert ";" in (synth_dir / "rides/syn_01.csv").read_text().splitlines()[0]


def test_basic_stats(synth_dir):
    ds = load_dataset(synth_dir)
    labels = pd.concat([df["label"] for df in ds.rides.values()])
    share = labels.value_counts(normalize=True)
    for lab in LABELS:
        assert share.get(lab, 0) > 0.02, (lab, share.to_dict())
    assert labels.isna().sum() == 0
    for rid, df in ds.rides.items():
        dt = np.median(np.diff(df["t_s"]))
        assert 50 <= 1 / dt <= 100.5
        rest = df[df["t_s"] <= 2.5][["accel_x", "accel_y", "accel_z"]].to_numpy()
        assert np.linalg.norm(rest.mean(axis=0)) == pytest.approx(9.81, abs=0.3), rid
        fix = df.dropna(subset=["latitude"])
        assert 0.8 <= len(fix) / df["t_s"].iloc[-1] <= 1.05
        assert fix["latitude"].between(BBOX["lat_min"] - 0.001, BBOX["lat_max"] + 0.001).all()
        assert fix["longitude"].between(BBOX["lon_min"] - 0.001, BBOX["lon_max"] + 0.001).all()


def test_pothole_peaks_exceed_smooth(synth_dir):
    ds = load_dataset(synth_dir)
    hb = [r for r in ds.rides if ds.meta.loc[r, "mount"] == "handlebar"]
    peaks = {"Pothole": [], "Smooth": []}
    for rid in hb:
        df = ds.rides[rid]
        dev = (np.linalg.norm(df[["accel_x", "accel_y", "accel_z"]], axis=1) - 9.81)
        win = (df["t_s"] // 1).astype(int)
        g = pd.DataFrame({"dev": np.abs(dev), "win": win, "label": df["label"]})
        for lab in peaks:
            peaks[lab] += g[g["label"] == lab].groupby("win")["dev"].max().tolist()
    assert np.mean(peaks["Pothole"]) > 3 * np.mean(peaks["Smooth"])


def test_help_lists_contract(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ["synth", "inspect", "normalize", "validate-labels", "train", "evaluate",
                "predict", "fetch-osm", "ingest", "serve", "report"]:
        assert cmd in out


def test_not_implemented(capsys):
    # `train` became real in P02; `fetch-osm` is still a stub until P03.
    assert main(["fetch-osm", "--tracks", "x"]) == 2
    assert "not implemented yet, phase P03" in capsys.readouterr().out


def test_inspect_smoke(synth_dir, capsys):
    assert main(["inspect", str(synth_dir / "rides/syn_00.csv")]) == 0
    out = capsys.readouterr().out
    for s in ["Native sampling rate", "GPS fixes", "First 5 s", "Pothole", "Warnings"]:
        assert s in out


def test_inspect_warns_without_gravity_or_gps(tmp_path, capsys):
    p = tmp_path / "lin.csv"
    p.write_text("timestamp_ms,accel_x,accel_y,accel_z\n" +
                 "\n".join(f"{i * 100},0.1,0,0.2" for i in range(80)) + "\n")
    assert main(["inspect", str(p)]) == 0
    out = capsys.readouterr().out
    assert "No GPS fix" in out and "Low sampling rate" in out and "expected ~9.81" in out


def test_normalize_roundtrip(synth_dir, tmp_path, capsys):
    src = synth_dir / "rides/syn_01.csv"  # semicolon + decimal comma, GPS on own rows
    out = tmp_path / "c.csv"
    assert main(["normalize", str(src), "--out", str(out), "--ride-id", "syn_01"]) == 0
    a, b = load_ride(src), load_ride(out)
    assert b.attrs["time_kind"] == "relative_ms"
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                  check_exact=False, atol=1e-6)


def test_cli_error_message(tmp_path, capsys):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    assert main(["inspect", str(p)]) == 1
    assert "Columns found" in capsys.readouterr().err
