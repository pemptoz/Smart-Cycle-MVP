"""No-leakage CV, rule baseline, model card and the CLI end-to-end on a tiny synthetic set."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from smartcycle.cli import main
from smartcycle.features.build import build_windows
from smartcycle.features.preprocess import PipelineConfig
from smartcycle.io import LABELS, load_dataset
from smartcycle.models.evaluation import make_splits
from smartcycle.models.rule import RuleBaseline, macro_f1_codes
from smartcycle.synth import generate


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    generate(d, n_rides=4, seed=5, minutes=2.0)
    return d


@pytest.fixture(scope="module")
def windows(tiny):
    return build_windows(load_dataset(tiny), PipelineConfig())


def test_grouped_folds_have_disjoint_rides(windows):
    meta = windows.meta
    sp = make_splits(meta, 1.0)
    assert sp.mode == "grouped" and len(sp.folds) == 4
    seen = set()
    for tr, te in sp.folds:
        tr_r, te_r = set(meta["ride_id"].iloc[tr]), set(meta["ride_id"].iloc[te])
        assert tr_r.isdisjoint(te_r)
        seen |= te_r
    assert seen == set(meta["ride_id"])  # every ride tested exactly once overall


def test_fallback_blocked_cv_with_two_rides(windows, capsys):
    meta = windows.meta[windows.meta["ride_id"].isin(["syn_00", "syn_01"])].reset_index(drop=True)
    sp = make_splits(meta, 1.0)
    assert sp.mode == "blocked-time" and "OPTIMISTIC" in sp.warning
    assert "OPTIMISTIC" in capsys.readouterr().err
    t0, t1 = meta["t_start"].to_numpy(), meta["t_end"].to_numpy()
    for tr, te in sp.folds:
        assert len(np.intersect1d(tr, te)) == 0
        for rid in meta["ride_id"].unique():
            a = te[meta["ride_id"].to_numpy()[te] == rid]
            b = tr[meta["ride_id"].to_numpy()[tr] == rid]
            if len(a) and len(b):
                assert not ((t1[b] > t0[a].min() - 2.0) & (t0[b] < t1[a].max() + 2.0)).any()


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 4, 500), rng.integers(0, 4, 500)
    assert macro_f1_codes(y, p, 4) == pytest.approx(f1_score(y, p, average="macro"))


def test_rule_baseline_learns_thresholds_on_training_data():
    n = 400
    rng = np.random.default_rng(0)
    y = np.array(["Smooth"] * n + ["Rough"] * n + ["Pothole"] * n + ["Irregularity"] * n)
    rms = np.r_[rng.uniform(0, 1, n), rng.uniform(2, 3, n), rng.uniform(2, 3, 2 * n)]
    peak_v = np.r_[rng.uniform(0, 2, 2 * n), rng.uniform(10, 12, n), rng.uniform(0, 2, n)]
    peak_h = np.r_[rng.uniform(0, 1, 3 * n), rng.uniform(5, 6, n)]
    X = pd.DataFrame({"a_vert__min": -peak_v, "a_vert__max": peak_v / 2,
                      "a_hmag__max": peak_h, "a_vert__rms": rms})
    m = RuleBaseline().fit(X, y)
    assert set(m.classes_) == set(LABELS)
    assert (m.predict(X) == y).mean() > 0.95
    assert m.predict_proba(X).sum(axis=1).tolist() == [1.0] * len(y)


def test_cli_end_to_end(tiny, tmp_path, capsys):
    reports, models = tmp_path / "reports", tmp_path / "models"
    assert main(["validate-labels", "--data-dir", str(tiny)]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "Windows per ride and class" in out

    assert main(["train", "--data-dir", str(tiny), "--out", str(models),
                 "--reports", str(reports)]) == 0
    out = capsys.readouterr().out
    assert "RandomForest" in out and "Rule baseline" in out and "SYNTHETIC" in out
    for f in ["metrics.json", "metrics.md", "confusion_matrix.png", "feature_importance.png",
              "signal_examples.png", "per_ride.csv"]:
        assert (reports / f).exists(), f
    metrics = json.loads((reports / "metrics.json").read_text())
    assert metrics["data_provenance"] == "synthetic" and metrics["cv_mode"] == "grouped"
    assert "SYNTHETIC" in (reports / "metrics.md").read_text()
    assert set(pd.read_csv(reports / "per_ride.csv")["data_provenance"]) == {"synthetic"}

    card = json.loads((models / "model_card.json").read_text())
    for k in ["data_provenance", "n_rides", "n_windows_per_class", "feature_set", "fs",
              "window_s", "step_s", "classes", "cv_metrics", "trained_at", "sklearn_version",
              "git_commit", "git_dirty", "data_dir", "cv_protocol", "augmentation",
              "n_synthetic_windows"]:
        assert k in card, k
    assert card["data_provenance"] == "synthetic" and card["n_rides"] == 4
    assert isinstance(card["git_dirty"], bool) and card["augmentation"] is None
    assert card["n_synthetic_windows"] == 0 and card["cv_protocol"]["cv"] == "grouped"
    assert set(card["classes"]) == set(LABELS)
    assert set(card["cv_metrics"]) == {"rule", "rf", "hgb"}
    assert sum(card["n_windows_per_class"].values()) == card["n_windows"]

    pred_csv = tmp_path / "pred.csv"
    assert main(["predict", str(tiny / "rides" / "syn_02.csv"), "--model",
                 str(models / "model.joblib"), "--out", str(pred_csv)]) == 0
    out = capsys.readouterr().out
    assert "trained on SYNTHETIC data" in out and "Events" in out
    pred = pd.read_csv(pred_csv)
    assert set(pred["pred"]) <= set(LABELS)
    assert pred.loc[pred["stationary"].astype(bool), "confidence"].isna().all()
    probs = pred[[f"p_{c}" for c in card["classes"]]].sum(axis=1)
    assert np.allclose(probs, 1.0)


def test_predict_without_model_fails_cleanly(tiny, tmp_path, capsys):
    assert main(["predict", str(tiny / "rides" / "syn_00.csv"), "--model",
                 str(tmp_path / "missing.joblib")]) == 1
    assert "run `python -m smartcycle train`" in capsys.readouterr().err
