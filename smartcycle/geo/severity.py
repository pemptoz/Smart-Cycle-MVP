"""Severity (1–5), segment score and heat band.

ALL OF THIS IS AN UNCALIBRATED HEURISTIC. Nothing here was validated against measured damage:
the bins map a peak acceleration to a 1–5 scale so the partner view can rank reports. Keep
`confidence` (model probability) separate from severity (signal amplitude).

Bins (P06) are FIXED by physical reasoning, NOT fitted to the ~17 real events:
- Pothole: peak |a_vert| (m/s², gravity removed) and Irregularity: peak a_hmag (horizontal
  magnitude, m/s²) use multiples of g as a convenient physical scale: 0.5 g, 1 g, 2 g, 3 g
  (4.9 / 9.8 / 19.6 / 29.4 m/s²). The Pothole 4/5 edge (29.4) falls between two near-identical
  real peaks (29.3 and 29.6): that split is not meaningful.
- Rough: mean window rms(a_vert) (m/s²) in bins [<0.315, 0.315–0.8, 0.8–1.25, 1.25–2, >=2],
  inspired by (not conformant with) the ISO 2631-1 comfort categories; ISO uses frequency-
  weighted rms at the seat over long exposures, we use unweighted rms at the handlebar.
- Segment score = sum of point-event severities / max(covered_m / 100, 1)
  + ROUGH_FULL_POINTS x rough_m / covered_m, i.e. "severity points per 100 m ridden" plus up to
  4 points for the share of the segment that is Rough (a fully Rough segment = "Needs
  attention"; only point events can push it to "Repair soon"). Band: >= 6 "Repair soon" (hi),
  >= 3 "Needs attention" (mid), else "Good" (lo) — same thresholds as the website.
"""

from __future__ import annotations

import math

SEVERITY_NOTE = ("Severity 1-5 is an uncalibrated heuristic derived from peak acceleration "
                 "bins; it is not a measured damage level.")

G = 9.81  # m/s²
_G_BINS = (0.5 * G, 1.0 * G, 2.0 * G, 3.0 * G)

SCORE_NOTE = ("Score = point-event severities per 100 m ridden + up to 4 points for the share "
              "of the segment that is Rough; >= 6 Repair soon, >= 3 Needs attention. "
              "Uncalibrated heuristic.")

# type -> (event field used, 4 ascending thresholds in m/s²). Uncalibrated heuristic.
SEVERITY_BINS: dict[str, tuple[str, tuple[float, float, float, float]]] = {
    "Pothole": ("peak_vert", _G_BINS),
    "Irregularity": ("peak_hmag", _G_BINS),
    "Rough": ("rms_vert", (0.315, 0.8, 1.25, 2.0)),
}
SCORE_CFG = {"rough_full_points": 4.0, "per_m": 100.0, "hi": 6.0, "mid": 3.0}
BANDS = {"hi": "Repair soon", "mid": "Needs attention", "lo": "Good"}


def severity_from_value(kind: str, value: float) -> int:
    """1..5 from the bins of `kind` (uncalibrated heuristic). NaN -> 1."""
    _, th = SEVERITY_BINS[kind]
    if value is None or not math.isfinite(value):
        return 1
    return 1 + sum(value >= t for t in th)


def severity(kind: str, event: dict) -> int:
    field, _ = SEVERITY_BINS[kind]
    return severity_from_value(kind, float(event.get(field, float("nan"))))


def peak_value(kind: str, event: dict) -> float:
    """The magnitude used for severity (reported as `peak`, m/s²)."""
    return float(event.get(SEVERITY_BINS[kind][0], float("nan")))


def segment_score(severity_sum: float, rough_m: float, covered_m: float) -> float:
    c = SCORE_CFG
    rough_share = min(rough_m / covered_m, 1.0) if covered_m > 0 else 0.0
    return (severity_sum / max(covered_m / c["per_m"], 1.0)
            + c["rough_full_points"] * rough_share)


def band(score: float) -> str:
    if score >= SCORE_CFG["hi"]:
        return "hi"
    if score >= SCORE_CFG["mid"]:
        return "mid"
    return "lo"
