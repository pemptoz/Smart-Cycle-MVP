"""Docs generated from the committed artifacts: model notes consistent with metrics / card,
committed docs up to date, unknown template keys rejected. Committed files only."""

import json
import re
from pathlib import Path

import pytest

from smartcycle.report.artifacts import load_artifacts
from smartcycle.report.render import render_all, render_template, values

ROOT = Path(__file__).resolve().parents[1]
# Numbers allowed in model_notes.json that are not artifact values (each one checked below).
RULE_CONSTANTS = {"10": "reporting rule: Pothole results are anecdotal below 10 events"}


@pytest.fixture(scope="module")
def art():
    return load_artifacts(ROOT)


def _artifact_numbers(art) -> set[str]:
    out = set()
    for k, v in values(art).items():
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", v):
            out.add(v.lstrip("+"))
            out.add(f"{float(v):.2f}")
    return out


def test_model_notes_match_metrics_and_card(art):
    notes, card, m = art["notes"], art["card"], art["metrics"]
    assert notes["data_provenance"] == card["data_provenance"] == m["data_provenance"]
    text = json.dumps({k: v for k, v in notes.items() if k != "written_for"})
    known = _artifact_numbers(art)
    for num in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text):
        assert num in known or num in RULE_CONSTANTS, \
            f"model_notes.json number {num} is not in reports/metrics.json / model_card.json"
    # the 'fewer than 10 events' claim must still be true
    if "fewer than 10" in text:
        assert m["models"]["rf"]["events"]["Pothole"]["n_labelled"] < 10
    if "Pocket test inconclusive" in text:
        assert m["pocket_test"]["invariant"]["n_windows"] < 10


def test_model_notes_check_detects_a_stale_number(art):
    known = _artifact_numbers(art)
    num = next(f"0.{i:03d}" for i in range(999, 0, -1) if f"0.{i:03d}" not in known)
    stale = dict(art, notes={**art["notes"], "headline": f"RF macro-F1 {num} on real data"})
    with pytest.raises(AssertionError):
        test_model_notes_match_metrics_and_card(stale)
    wrong = dict(art, notes={**art["notes"], "data_provenance": "synthetic"})
    with pytest.raises(AssertionError):
        test_model_notes_match_metrics_and_card(wrong)


def test_committed_docs_are_up_to_date(art):
    stale = [p for p, text in render_all(art, ROOT).items()
             if not (ROOT / p).exists() or (ROOT / p).read_text(encoding="utf-8") != text]
    assert not stale, f"run `python -m smartcycle report`: stale {stale}"


def test_results_md_numbers_come_from_artifacts(art):
    text = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    rf = art["metrics"]["models"]["rf"]
    assert f"{rf['macro_f1']:.3f}" in text
    assert f"{art['insample']['macro_f1']:.3f}" in text
    assert art["metrics"]["data_provenance"].upper() in text
    assert "Limitations and honest caveats" in text


def test_unknown_template_key_fails(art):
    vals = values(art)
    assert render_template("RF {{metrics.models.rf.macro_f1}}", vals) == \
        f"RF {art['metrics']['models']['rf']['macro_f1']:.3f}"
    with pytest.raises(KeyError, match="no.such"):
        render_template("{{no.such.key}}", vals)
