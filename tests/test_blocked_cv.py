"""Blocked + purged CV for the tiny real dataset (each recording ~ one class)."""

import numpy as np
import pandas as pd
import pytest

from smartcycle.cli import main
from smartcycle.models.evaluation import BLOCKED_HEADER, block_ids, make_splits


def one_class_per_ride(durations=None, window_s=1.0, step_s=0.5) -> pd.DataFrame:
    """Like the real data: ride bump = Pothole, irregs = Irregularity, rug = Rough,
    smooth = Smooth (each ride ONE class)."""
    durations = durations or {"bump": 55, "irregs": 60, "rug": 36, "smooth": 35}
    cls = {"bump": "Pothole", "irregs": "Irregularity", "rug": "Rough", "smooth": "Smooth"}
    rows = []
    for rid, d in durations.items():
        for t in np.arange(0, d - window_s + 1e-9, step_s):
            rows.append({"ride_id": rid, "t_start": t, "t_end": t + window_s,
                         "label": cls.get(rid, "Smooth")})
    return pd.DataFrame(rows)


def test_blocked_folds_purged_and_all_classes_in_every_test_fold(capsys):
    from smartcycle.models import evaluation
    evaluation._WARNED.clear()
    meta = one_class_per_ride()
    sp = make_splits(meta, 1.0, cv="blocked", block_s=15.0, purge_s=2.0)
    assert sp.mode == "blocked" and BLOCKED_HEADER in capsys.readouterr().err
    assert sp.protocol["n_splits"] == 3  # rug/smooth have 3 blocks of 15 s -> k = 3
    y = meta["label"].to_numpy()
    rides = meta["ride_id"].to_numpy()
    t0, t1 = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
    groups = block_ids(meta, 15.0)
    tested = np.zeros(len(meta), int)
    for tr, te in sp.folds:
        assert set(y[te]) == set(y)  # every class in every test fold
        assert len(np.intersect1d(tr, te)) == 0
        for rid in np.unique(rides[te]):
            a, b = te[rides[te] == rid], tr[rides[tr] == rid]
            gaps = np.minimum.reduce([np.abs(t0[b][:, None] - t1[a][None, :]),
                                      np.abs(t1[b][:, None] - t0[a][None, :])])
            overl = (t1[b][:, None] > t0[a][None, :]) & (t0[b][:, None] < t1[a][None, :])
            assert not overl.any() and (gaps >= 2.0 - 1e-9).all()
        tested[te] += 1
    assert (tested == 1).all()  # every window tested exactly once
    # no test block is split between train and test
    for tr, te in sp.folds:
        assert not set(groups[te]) & set(groups[tr])


def test_blocked_refuses_when_a_class_has_one_block():
    meta = one_class_per_ride({"bump": 40, "irregs": 40, "rug": 12, "smooth": 40})
    with pytest.raises(ValueError, match="at least 2 blocks per class"):
        make_splits(meta, 1.0, cv="blocked", block_s=15.0)


@pytest.mark.parametrize("cv", ["logo", "grouped"])
def test_ride_level_cv_refuses_when_classes_live_in_one_ride(cv):
    meta = one_class_per_ride({"bump": 30, "irregs": 30, "rug": 30, "smooth": 30})
    with pytest.raises(ValueError, match="fewer than 2 rides.*Pothole only in \\['bump'\\]"):
        make_splits(meta, 1.0, cv=cv)


def test_logo_folds_leave_one_ride_out():
    meta = one_class_per_ride({"a": 20, "b": 20, "c": 20})
    meta["label"] = np.where(meta["t_start"] < 10, "Smooth", "Rough")
    sp = make_splits(meta, 1.0, cv="logo")
    assert len(sp.folds) == 3
    for tr, te in sp.folds:
        assert meta["ride_id"].iloc[te].nunique() == 1
        assert not set(meta["ride_id"].iloc[tr]) & set(meta["ride_id"].iloc[te])


def test_cli_logo_refusal_message(tmp_path, capsys):
    """One synthetic ride per class is not available offline; the CLI path is exercised by
    the evaluation tests: here we only check that the parser accepts the new options."""
    from smartcycle.cli import build_parser
    a = build_parser().parse_args(["evaluate", "--data-dir", str(tmp_path), "--cv", "blocked",
                                   "--block-s", "10", "--purge-s", "3", "--mount", "handlebar",
                                   "--augment"])
    assert (a.cv, a.block_s, a.purge_s, a.mount, a.augment) == ("blocked", 10.0, 3.0,
                                                                 "handlebar", True)
    assert main(["evaluate", "--data-dir", str(tmp_path), "--cv", "logo"]) == 1
    assert "rides.csv" in capsys.readouterr().err


def test_experiments_speed_confound_and_pocket_test(tmp_path, capsys):
    """(i)-(iv) on a small SYNTHETIC set with both mounts, blocked CV, handlebar only in CV."""
    from smartcycle.features.build import build_windows
    from smartcycle.features.preprocess import PipelineConfig
    from smartcycle.io import load_dataset
    from smartcycle.io.dataset import select_mount
    from smartcycle.models.evaluation import CVConfig, run_ablations
    from smartcycle.synth import generate

    generate(tmp_path, n_rides=6, seed=4, minutes=1.0)
    full = load_dataset(tmp_path)
    assert set(full.meta["mount"]) == {"handlebar", "pocket"}
    cfg = PipelineConfig()
    base = build_windows(select_mount(full, "handlebar"), cfg)
    ab = run_ablations(full, cfg, base, cvc=CVConfig(cv="blocked"), mount="handlebar")
    names = list(ab["ablation"])
    for key in ("WITHOUT speed features", "speed features ONLY", "DIAGNOSTIC 3 classes", "invariant+fixed", "2 s windows",
                "train handlebar -> test pocket (invariant)",
                "train handlebar -> test pocket (invariant+fixed)"):
        assert any(key in n for n in names), key
    spd = {r["label"]: r for r in ab.attrs["speed_by_class"]}
    assert set(spd) <= {"Smooth", "Rough", "Pothole", "Irregularity"} and spd["Smooth"]["n"] > 0
    pk = ab.attrs["pocket"]["invariant"]
    assert pk["classes_in_pocket"] and set(pk["per_class"]) == set(pk["classes_in_pocket"])
    assert "macro_f1" not in pk  # no overall score for the pocket test
    assert sum(v["n_windows"] for v in pk["per_class"].values()) == pk["n_windows"]
    th = ab.attrs["three_class"]
    assert th["classes"] == ["Smooth", "Rough", "Impact"]
    assert "/" in th["events"]["Impact"]["detected_of"]
    # event results are raw counts "k/n", never rates
    assert all(isinstance(v, str) and "/" in v for v in ab["pothole_events"].dropna())
