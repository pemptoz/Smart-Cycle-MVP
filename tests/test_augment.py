"""Augmentation is applied to TRAINING folds only and never derives from test samples."""

import numpy as np
import pytest

from smartcycle.features.augment import augment_training, transform_window, window_len
from smartcycle.features.build import build_windows
from smartcycle.features.features import INVARIANT, features_from_arrays
from smartcycle.features.preprocess import PipelineConfig, channels_from
from smartcycle.io import LABELS, load_dataset
from smartcycle.models.evaluation import CVConfig, cross_validate, make_splits
from smartcycle.synth import generate


@pytest.fixture(scope="module")
def wt(tmp_path_factory):
    d = tmp_path_factory.mktemp("aug")
    generate(d, n_rides=3, seed=21, minutes=1.0)
    return build_windows(load_dataset(d), PipelineConfig())


def test_no_augmented_window_derives_from_a_test_block(wt):
    cfg = PipelineConfig()
    sp = make_splits(wt.meta, cfg.window_s, cv="blocked", block_s=15.0, purge_s=2.0)
    meta = wt.meta
    for f, (tr, te) in enumerate(sp.folds):
        aug = augment_training(wt, tr, cfg, seed=f, exclude=te, return_sources=True)
        assert len(aug.y) >= len(tr) and "paste" in aug.kind
        assert set(aug.y) <= set(LABELS)
        assert len(aug.sources) == len(aug.y) == len(aug.X)
        # samples of test windows, and samples of training windows, per ride
        test_s, train_s = {}, {}
        for sel, dst in ((te, test_s), (tr, train_s)):
            for k in sel:
                rid, i0 = meta["ride_id"].iloc[k], int(meta["i0"].iloc[k])
                w = window_len(cfg, wt.preps[rid].fs)
                dst.setdefault(rid, set()).update(range(i0, i0 + w))
        for srcs in aug.sources:
            for rid, idx in srcs:
                used = set(int(i) for i in idx)
                assert used <= train_s[rid], "augmented window uses non-training samples"
                assert not used & test_s.get(rid, set()), "augmented window uses test samples"


def test_cross_validate_with_augmentation_keeps_test_folds_real(wt):
    cfg = PipelineConfig()
    res, sp, _ = cross_validate(wt, cfg, models=("rf",), cvc=CVConfig(cv="blocked",
                                                                       augment=True))
    oof = res["rf"].oof_pred
    assert len(oof) == len(wt.meta)  # predictions only for the original (real) windows
    assert all(n > 0 for n in sp.protocol["n_extra_train_windows_per_fold"])


def test_rotation_leaves_invariant_channels_unchanged(wt):
    cfg = PipelineConfig()
    rid = wt.meta["ride_id"].iloc[0]
    prep = wt.preps[rid]
    w = window_len(cfg, prep.fs)
    i0 = int(wt.meta["i0"].iloc[len(wt.meta) // 3])
    allowed = np.ones(len(prep.t), bool)
    g_mean = float(np.linalg.norm(np.column_stack([prep.signals[c] for c in ("gx", "gy", "gz")]),
                                  axis=1).mean())
    acc = np.column_stack([prep.signals[a] + prep.signals[g]
                           for a, g in (("ax", "gx"), ("ay", "gy"), ("az", "gz"))])
    mag_mean = float(np.linalg.norm(acc, axis=1).mean())
    dyn0, g0, _, f0 = transform_window(prep, i0, w, allowed, np.random.default_rng(0),
                                       warp=False, rotate=False, scale=False, noise=False)
    dyn1, g1, _, _ = transform_window(prep, i0, w, allowed, np.random.default_rng(1),
                                      warp=False, rotate=True, scale=False, noise=False)
    c0 = channels_from(dyn0, g0, g_mean, mag_mean)
    c1 = channels_from(dyn1, g1, g_mean, mag_mean)
    # identity transform reproduces the real window's channels exactly
    for ch in ("a_vert", "a_hmag", "a_dev", "ax"):
        assert np.allclose(c0[ch], prep.signals[ch][i0:i0 + w], atol=1e-9)
    for ch in INVARIANT:
        assert np.allclose(c0[ch], c1[ch], atol=1e-6)
    assert not np.allclose(c0["ax"], c1["ax"])  # fixed axes do change
    assert f0 == 1.0


def test_identity_features_match_real_window_features(wt):
    prep = next(iter(wt.preps.values()))
    rid = prep.ride_id
    k = np.flatnonzero(wt.meta["ride_id"].to_numpy() == rid)[:5]
    w = window_len(PipelineConfig(), prep.fs)
    idx = wt.meta["i0"].to_numpy()[k][:, None] + np.arange(w)[None, :]
    chans = {c: prep.signals[c][idx] for c in ("a_vert", "a_hmag", "a_dev", "ax", "ay", "az")}
    X = features_from_arrays(chans, prep.fs, wt.meta["speed"].to_numpy()[k])
    assert np.allclose(X.to_numpy(), wt.X.iloc[k][list(X.columns)].to_numpy(), equal_nan=True)


def test_training_provenance_rules():
    from smartcycle.models.evaluation import training_provenance
    assert training_provenance("real", CVConfig()) == "real"
    assert training_provenance("real", CVConfig(augment=True)) == "mixed"
    assert training_provenance("real", CVConfig(extra_train="calibrated")) == "mixed"
    assert training_provenance("synthetic", CVConfig(augment=True)) == "synthetic"
