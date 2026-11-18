"""Provenance fails closed (P90 review, ML-2): a directory is real only when its MANIFEST.json
says so; synthetic tags anywhere win; train / evaluate refuse unknown provenance."""

import json

import pytest

from smartcycle.cli import main
from smartcycle.io.dataset import load_dataset, read_provenance, resolve_provenance
from smartcycle.synth import generate


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn")
    generate(d, n_rides=4, seed=5, minutes=1.0)
    return d


def strip(d, tmp_path, name="nomanifest", notes=True):
    """Copy of a synthetic dir without MANIFEST.json (and optionally without synthetic tags)."""
    import shutil
    out = tmp_path / name
    shutil.copytree(d, out)
    (out / "MANIFEST.json").unlink()
    if not notes:
        p = out / "rides.csv"
        p.write_text(p.read_text().replace("synthetic", "phone").replace("SYNTHETIC", "x"))
    return out


def test_synthetic_without_manifest_is_still_synthetic(syn, tmp_path):
    """The reviewer's reproduction: removing MANIFEST.json used to turn synthetic into real."""
    d = strip(syn, tmp_path)
    assert read_provenance(syn) == "synthetic"
    assert read_provenance(d) == "synthetic"  # rides.csv device / notes say synthetic
    assert load_dataset(d).provenance == "synthetic"


def test_untagged_directory_is_unknown_not_real(syn, tmp_path):
    d = strip(syn, tmp_path, notes=False)
    assert read_provenance(d) == "unknown"
    (d / "MANIFEST.json").write_text(json.dumps({"provenance": "real"}))
    assert read_provenance(d) == "real"
    (d / "MANIFEST.json").write_text("{not json")
    assert read_provenance(d) == "unknown"


def test_override_cannot_turn_synthetic_into_real(syn, tmp_path):
    with pytest.raises(ValueError, match="synthetic"):
        resolve_provenance(syn, "real")
    d = strip(syn, tmp_path, notes=False)
    assert resolve_provenance(d, "real") == "real"
    assert resolve_provenance(d, None) == "unknown"


def test_evaluate_refuses_unknown_provenance(syn, tmp_path, capsys):
    d = strip(syn, tmp_path, notes=False)
    rep = tmp_path / "rep"
    assert main(["evaluate", "--data-dir", str(d), "--reports", str(rep)]) == 1
    assert "provenance is unknown" in capsys.readouterr().err
    assert not rep.exists()
    assert main(["evaluate", "--data-dir", str(syn), "--reports", str(rep),
                 "--provenance", "real"]) == 1
    assert "refused" in capsys.readouterr().err


def test_committed_real_data_is_declared_real():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "data" / "real_public"
    assert json.loads((root / "MANIFEST.json").read_text())["provenance"] == "real"
    assert read_provenance(root) == "real"
