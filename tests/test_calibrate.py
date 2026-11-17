"""Calibrated synthetic windows: training-fold only, real-only test folds, MANIFEST provenance."""

import json

import numpy as np

from smartcycle.cli import main
from smartcycle.features.build import build_windows
from smartcycle.features.preprocess import PipelineConfig
from smartcycle.io import LABELS, load_dataset
from smartcycle.io.dataset import read_provenance
from smartcycle.models.evaluation import CVConfig, cross_validate, make_splits
from smartcycle.synth import generate
from smartcycle.synth.calibrate import calibrate, calibrated_windows


def test_calibrated_windows_use_training_blocks_only(tmp_path):
    generate(tmp_path, n_rides=3, seed=8, minutes=1.0)
    wt = build_windows(load_dataset(tmp_path), PipelineConfig())
    cfg = PipelineConfig()
    sp = make_splits(wt.meta, 1.0, cv="blocked")
    tr, te = sp.folds[0]
    cal = calibrate(wt, tr, cfg)
    # calibration statistics must not change when the TEST windows are altered
    for rid, p in wt.preps.items():
        for k in te[wt.meta["ride_id"].to_numpy()[te] == rid]:
            i0 = int(wt.meta["i0"].iloc[k])
            for c in ("ax", "ay", "az"):
                p.signals[c][i0:i0 + 50] *= 10.0
    cal2 = calibrate(wt, tr, cfg)
    for c in cal.std:
        assert np.allclose(cal.std[c], cal2.std[c])
    assert {k: len(v) for k, v in cal.events.items()} == {k: len(v) for k, v in cal2.events.items()}
    X, y = calibrated_windows(wt, tr, cfg, seed=0)
    assert list(X.columns) == list(wt.X.columns) and set(y) <= set(LABELS)
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    res, sp2, _ = cross_validate(wt, cfg, models=("rf",),
                                 cvc=CVConfig(cv="blocked", extra_train="calibrated"))
    assert len(res["rf"].oof_pred) == len(wt.meta)  # test folds: real windows only
    assert all(n > 0 for n in sp2.protocol["n_extra_train_windows_per_fold"])


def test_cli_synth_calibrate_writes_manifest(tmp_path, capsys):
    real = tmp_path / "real"
    generate(real, n_rides=3, seed=9, minutes=1.0)
    (real / "MANIFEST.json").unlink()  # pretend real (provenance only matters for the output)
    out = tmp_path / "cal"
    assert main(["synth", "--calibrate-from", str(real), "--labels", str(real / "labels.csv"),
                 "--out", str(out), "--rides", "2", "--minutes", "0.5"]) == 0
    m = json.loads((out / "MANIFEST.json").read_text())
    assert m["provenance"] == "synthetic-calibrated" and read_provenance(out) == "synthetic"
    assert "SYNTHETIC-CALIBRATED" in capsys.readouterr().out
    ds = load_dataset(out)
    assert len(ds.rides) == 2 and ds.provenance == "synthetic"
