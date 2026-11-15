"""Severity (1–5), segment score and heat band.

ALL OF THIS IS AN UNCALIBRATED HEURISTIC. Nothing here was validated against measured damage:
the bins map a peak acceleration to a 1–5 scale so the partner view can rank reports. Keep
`confidence` (model probability) separate from severity (signal amplitude).

- Pothole: peak |a_vert| (m/s², gravity removed) in bins [<6, 6–9, 9–13, 13–18, >=18].
- Irregularity: peak a_hmag (horizontal magnitude, m/s²) in bins [<3, 3–5, 5–8, 8–12, >=12].
- Rough: mean window rms(a_vert) (m/s²) in bins [<0.5, 0.5–0.8, 0.8–1.2, 1.2–2, >=2]
  (picked from the SYNTHETIC Rough distribution: 10/50/90 % = 0.41/0.67/1.15; revisit on
  real data).
- Segment score = (sum of point-event severities + rough_length_m / 25) / max(covered_m / 100, 1),
  i.e. "severity points per 100 m ridden". Band: >= 6 "Repair soon" (hi), >= 3 "Needs
  attention" (mid), else "Good" (lo) — same thresholds as the website.
"""

from __future__ import annotations

import math

SEVERITY_NOTE = ("Severity 1-5 is an uncalibrated heuristic derived from peak acceleration "
                 "bins; it is not a measured damage level.")

# type -> (event field used, 4 ascending thresholds)
SEVERITY_BINS: dict[str, tuple[str, tuple[float, float, float, float]]] = {
    "Pothole": ("peak_vert", (6.0, 9.0, 13.0, 18.0)),
    "Irregularity": ("peak_hmag", (3.0, 5.0, 8.0, 12.0)),
    "Rough": ("rms_vert", (0.5, 0.8, 1.2, 2.0)),
}
SCORE_CFG = {"rough_m_per_point": 25.0, "per_m": 100.0, "hi": 6.0, "mid": 3.0}
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
    return (severity_sum + rough_m / c["rough_m_per_point"]) / max(covered_m / c["per_m"], 1.0)


def band(score: float) -> str:
    if score >= SCORE_CFG["hi"]:
        return "hi"
    if score >= SCORE_CFG["mid"]:
        return "mid"
    return "lo"
