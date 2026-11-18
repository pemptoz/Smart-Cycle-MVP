"""The final REAL model is versioned (models/model.joblib): a fresh clone loads it, it is the
model described by models/model_card.json and reports/insample.json, and synthetic runs never
overwrite it. Committed files only, no network."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import sklearn

from smartcycle.models.card import ModelLoadError, load_model

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "model.joblib"
CARD = ROOT / "models" / "model_card.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


@pytest.fixture(scope="module")
def card() -> dict:
    return json.loads(CARD.read_text())


def test_committed_model_is_tracked_and_others_are_ignored():
    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        pytest.skip("not a git checkout")
    assert git("ls-files", "--error-unmatch", "models/model.joblib").returncode == 0, \
        "models/model.joblib must be committed (a fresh clone needs it)"
    for p in ("models/synthetic/model.joblib", "models/other.joblib", "reports/x.joblib"):
        assert git("check-ignore", "-q", p).returncode == 0, f"{p} must stay gitignored"
    assert git("check-ignore", "-q", "models/model.joblib").returncode == 1
    assert MODEL.stat().st_size < 10 * 1024 * 1024


def test_committed_model_matches_card_and_installed_sklearn(card):
    b = load_model(MODEL)  # strict: version + structure checks
    assert card["sklearn_version"] == sklearn.__version__
    pin = next(line for line in (ROOT / "pyproject.toml").read_text().splitlines()
               if "scikit-learn" in line)
    assert f'scikit-learn=={card["sklearn_version"]}' in pin
    assert b["model_name"] == card["model"] and b["data_provenance"] == card["data_provenance"]
    assert card["data_provenance"] == "real" and card["git_dirty"] is False
    assert len(b["feature_columns"]) == card["n_features"]
    assert list(b["classes"]) == card["classes"]
    assert not [c for c in b["feature_columns"] if "speed" in c]  # speed excluded
    assert card["include_speed"] is False


def test_committed_model_reproduces_reported_insample_metrics(card):
    """Same model as reports/insample.json (computed by `report` from the saved model)."""
    from smartcycle.report.artifacts import compute_insample

    want = json.loads((ROOT / "reports" / "insample.json").read_text())
    got = compute_insample(ROOT / "data" / "real_public", MODEL, card)
    assert got["n_windows"] == want["n_windows"] == card["n_windows"]
    assert got["macro_f1"] == pytest.approx(want["macro_f1"], abs=1e-12)


def test_load_model_refuses_corrupt_and_mismatched_files(tmp_path, card):
    bad = tmp_path / "model.joblib"
    bad.write_bytes(b"garbage")
    with pytest.raises(ModelLoadError, match="Reproduce"):
        load_model(bad)
    shutil.copyfile(MODEL, bad)
    (tmp_path / "model_card.json").write_text(json.dumps({**card, "sklearn_version": "0.1"}))
    with pytest.raises(ModelLoadError, match="scikit-learn 0.1"):
        load_model(bad)
    with pytest.raises(ModelLoadError, match="not found"):
        load_model(tmp_path / "missing.joblib")


def test_synthetic_training_never_overwrites_the_real_model(tmp_path, monkeypatch):
    """`train` on SYNTHETIC data, run from a copy of the repo layout: artifacts go to
    models/synthetic/ and reports/synthetic/, the real models/model.joblib is untouched."""
    from smartcycle.cli import main
    from smartcycle.synth import generate

    work = tmp_path / "repo"
    (work / "models").mkdir(parents=True)
    shutil.copyfile(MODEL, work / "models" / "model.joblib")
    shutil.copyfile(CARD, work / "models" / "model_card.json")
    before = sha(work / "models" / "model.joblib"), sha(work / "models" / "model_card.json")
    data = tmp_path / "syn"
    generate(data, n_rides=4, seed=3, minutes=1.0)
    monkeypatch.chdir(work)
    assert main(["train", "--data-dir", str(data), "--final-model", "rule"]) == 0
    assert (work / "models" / "synthetic" / "model.joblib").exists()
    assert (work / "reports" / "synthetic" / "metrics.json").exists()
    assert not (work / "reports" / "metrics.json").exists()
    assert (sha(work / "models" / "model.joblib"),
            sha(work / "models" / "model_card.json")) == before
    syn = json.loads((work / "models" / "synthetic" / "model_card.json").read_text())
    assert syn["data_provenance"] == "synthetic"
