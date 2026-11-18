"""Event-level metric (P90 review, ML-1): one-to-one matching on the peak, long predicted events
cut into 2 s units, so degenerate predictors can no longer look perfect."""

import numpy as np
import pandas as pd

from smartcycle.features.preprocess import PipelineConfig
from smartcycle.geo.events import event_units, match_events
from smartcycle.models.evaluation import event_metrics


def windows(n_s: float, pred, step=0.5, win=1.0, peaks=None):
    """1 s windows every 0.5 s over n_s seconds; `pred` = class or list; peak_mag optional."""
    t0 = np.arange(0, n_s - win + 1e-9, step)
    p = pred if isinstance(pred, list) else [pred] * len(t0)
    pm = peaks if peaks is not None else np.ones(len(t0))
    return pd.DataFrame({"t_start": t0, "t_end": t0 + win, "t_peak": t0 + win / 2,
                         "pred": p, "peak_mag": pm, "stationary": False, "speed": 4.0,
                         "ride_id": "r", "confidence": 1.0})


TRUTH = [(10.0, 10.5, "Irregularity"), (20.0, 20.5, "Irregularity"),
         (30.0, 30.4, "Irregularity")]


def test_long_predicted_event_is_cut_into_units():
    w = windows(60, "Irregularity")
    units = event_units(w, "Irregularity")
    assert len(units) == 30  # 60 s of predictions = 30 units of 2 s, not one event
    assert all(u[1] - u[0] <= 3.0 for u in units)  # chunk of window starts < 2 s, + 1 s window


def test_whole_ride_predictor_has_many_false_alarms():
    """The reviewer's case: predicting Irregularity everywhere used to give 3/3 with 0 FA."""
    w = windows(60, "Irregularity")
    m, n, k = match_events(TRUTH, event_units(w, "Irregularity"), "Irregularity")
    assert (m, n) == (3, 3) and k == 30 and k - m == 27  # recall 1 but precision 0.1
    cfg = PipelineConfig()
    new = event_metrics(w, w["pred"].to_numpy(), w["confidence"].to_numpy(),
                        {"r": TRUTH}, cfg)["Irregularity"]
    old = event_metrics(w, w["pred"].to_numpy(), w["confidence"].to_numpy(),
                        {"r": TRUTH}, cfg, legacy=True)["Irregularity"]
    assert old["detected_of"] == "3/3" and old["false_alarms"] == 0  # the lenient metric
    assert new["detected_of"] == "3/3" and new["false_alarms"] == 27
    assert abs(new["event_precision"] - 0.1) < 1e-9


def test_all_smooth_detects_nothing():
    w = windows(60, "Smooth")
    m = event_metrics(w, w["pred"].to_numpy(), w["confidence"].to_numpy(), {"r": TRUTH},
                      PipelineConfig())["Irregularity"]
    assert m["detected"] == 0 and m["n_predicted"] == 0 and m["event_precision"] is None


def test_one_to_one_matching():
    truth = [(10.0, 10.5, "Pothole"), (11.0, 11.5, "Pothole")]
    # one predicted unit between two labelled events: matches only one of them
    assert match_events(truth, [(10.0, 11.5, 10.8)], "Pothole") == (1, 2, 1)
    # two units on one labelled event: one match, one false alarm
    assert match_events(truth[:1], [(9.5, 10.5, 10.1), (10.0, 11.0, 10.6)], "Pothole") == \
        (1, 1, 2)
    # both events matched by their own unit, closest pairs first
    assert match_events(truth, [(9.5, 10.5, 10.2), (10.5, 11.5, 11.2)], "Pothole") == (2, 2, 2)
    # peak outside the tolerance: no match
    assert match_events(truth[:1], [(12.0, 13.0, 12.6)], "Pothole") == (0, 1, 1)
    # other classes are ignored
    assert match_events(truth, [(10.0, 11.0, 10.2)], "Irregularity") == (0, 0, 1)


def test_unit_peak_is_the_strongest_window():
    pm = np.ones(4)  # windows start at 0, 0.5, 1, 1.5 s: one 2 s unit
    pm[2] = 9.0
    w = windows(2.5, "Pothole", peaks=pm)
    (u,) = event_units(w, "Pothole")
    assert u[2] == w["t_peak"].iloc[2]


def test_stationary_windows_never_count():
    w = windows(10, "Pothole")
    w["stationary"] = True
    assert event_units(w, "Pothole") == []
