import pandas as pd

from smartcycle.geo.events import event_recall, false_alarms, merge_events


def W(rows):
    return pd.DataFrame(rows, columns=["t_start", "t_end", "pred", "confidence", "peak_mag",
                                       "t_peak", "lat", "lon", "stationary"])


def test_merge_consecutive_point_windows_and_gap():
    w = W([(0.0, 1.0, "Pothole", 0.8, 5.0, 0.5, 1.0, 1.0, False),
           (0.5, 1.5, "Pothole", 0.6, 9.0, 1.2, 2.0, 2.0, False),
           (1.0, 2.0, "Smooth", 0.9, 1.0, 1.5, 3.0, 3.0, False),
           (2.0, 3.0, "Pothole", 0.7, 4.0, 2.5, 4.0, 4.0, False),   # gap 0.5 s -> merged
           (5.0, 6.0, "Pothole", 0.9, 6.0, 5.5, 5.0, 5.0, False)])  # gap 2 s -> new event
    ev = merge_events(w, gap_s=1.0)
    assert list(ev["label"]) == ["Pothole", "Pothole"]
    first = ev.iloc[0]
    assert (first["t_start"], first["t_end"], first["n_windows"]) == (0.0, 3.0, 3)
    assert first["peak_mag"] == 9.0 and first["lat"] == 2.0 and first["t_peak"] == 1.2
    assert abs(first["mean_confidence"] - 0.7) < 1e-9


def test_rough_stretch_and_stationary_ignored():
    w = W([(0.0, 1.0, "Rough", 0.8, 2.0, 0.5, 1.0, 1.0, False),
           (0.5, 1.5, "Rough", 0.8, 2.5, 1.0, 1.0, 1.0, False),
           (1.0, 2.0, "Irregularity", 0.9, 5.0, 1.5, 1.0, 1.0, True),  # stationary
           (2.0, 3.0, "Smooth", 0.9, 1.0, 2.5, 1.0, 1.0, False)])
    ev = merge_events(w)
    assert list(ev["kind"]) == ["stretch"] and ev.iloc[0]["t_end"] == 1.5


def test_empty():
    assert merge_events(W([])).empty


def test_event_recall_and_false_alarms():
    truth = [(10.0, 11.0, "Pothole"), (30.0, 31.0, "Pothole"), (50.0, 51.0, "Irregularity")]
    w = W([(11.5, 12.5, "Pothole", 0.9, 5, 12, 0, 0, False),   # within +-1 s of 1st
           (40.0, 41.0, "Pothole", 0.9, 5, 40.5, 0, 0, False)])  # false alarm
    assert event_recall(truth, w, "Pothole") == (1, 2)
    assert event_recall(truth, w, "Irregularity") == (0, 1)
    ev = merge_events(w)
    assert false_alarms(truth, ev, "Pothole") == 1
