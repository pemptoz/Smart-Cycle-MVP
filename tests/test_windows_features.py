import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from smartcycle.features.build import ride_windows
from smartcycle.features.features import FIXED, INVARIANT, select_columns
from smartcycle.features.preprocess import PipelineConfig, preprocess_ride
from smartcycle.features.windows import make_windows, window_label
from smartcycle.io import load_dataset

RUNS = [(0.0, 10.0, "Smooth"), (10.0, 20.0, "Rough"), (4.0, 5.0, "Pothole"),
        (4.5, 6.0, "Irregularity"), (14.0, 15.0, "Irregularity")]


def test_event_overlap_threshold():
    assert window_label(3.7, 4.7, RUNS) == "Pothole"  # overlap 0.7 s
    assert window_label(3.25, 4.25, [(0, 10, "Smooth"), (4.0, 5.0, "Pothole")]) == "Smooth"
    assert window_label(3.3, 4.3, [(0, 10, "Smooth"), (4.0, 5.0, "Pothole")]) == "Pothole"


def test_pothole_has_priority_over_irregularity():
    assert window_label(4.5, 5.5, RUNS) == "Pothole"  # both overlap >= 0.3 s
    assert window_label(5.0, 6.0, RUNS) == "Irregularity"


def test_majority_rough_smooth_and_uncovered():
    assert window_label(9.0, 10.0, RUNS) == "Smooth"
    assert window_label(9.6, 10.6, RUNS) == "Rough"
    assert window_label(25.0, 26.0, RUNS) is None  # no label coverage
    assert window_label(19.8, 20.8, RUNS) is None  # coverage 0.2 s < 50 %


@pytest.fixture(scope="module")
def syn_ride(tmp_path_factory):
    from smartcycle.synth import generate
    d = tmp_path_factory.mktemp("synw")
    generate(d, n_rides=2, seed=3, minutes=1.0)
    ds = load_dataset(d)
    return ds.rides["syn_00"]


def test_windows_are_labelled_and_regular(syn_ride):
    cfg = PipelineConfig()
    prep = preprocess_ride(syn_ride, cfg, "syn_00")
    meta, idx = make_windows(prep, cfg, labelled=True)
    assert len(meta) > 50 and meta["label"].notna().all()
    assert idx.shape[1] == 50
    assert np.allclose(meta["t_end"] - meta["t_start"], 1.0)
    assert prep.valid[idx].all()  # no window crosses a sampling gap
    meta_u, _ = make_windows(prep, cfg, labelled=False)
    assert len(meta_u) >= len(meta)


def test_orientation_invariance(syn_ride):
    """A constant random rotation of the phone must not change the `invariant` features."""
    cfg = PipelineConfig()
    rot = syn_ride.copy()
    R = Rotation.from_euler("xyz", [37, -112, 64], degrees=True)
    rot[["accel_x", "accel_y", "accel_z"]] = R.apply(
        syn_ride[["accel_x", "accel_y", "accel_z"]].to_numpy())
    _, X0 = ride_windows(preprocess_ride(syn_ride, cfg, "a"), cfg, labelled=True)
    _, X1 = ride_windows(preprocess_ride(rot, cfg, "a"), cfg, labelled=True)
    inv = [c for c in X0.columns if c.split("__")[0] in INVARIANT]
    np.testing.assert_allclose(X1[inv].to_numpy(), X0[inv].to_numpy(), rtol=1e-5, atol=1e-6)
    fixed = [c for c in X0.columns if c.split("__")[0] in FIXED]
    assert not np.allclose(X1[fixed].to_numpy(), X0[fixed].to_numpy(), rtol=1e-3, atol=1e-3)


def test_feature_selection():
    cols = ["a_vert__rms", "ax__rms", "speed", "speed_missing", "rms_vert_per_speed",
            "peak_ratio_h_v"]
    assert select_columns(cols, "invariant", True) == ["a_vert__rms", "speed", "speed_missing",
                                                       "rms_vert_per_speed", "peak_ratio_h_v"]
    assert select_columns(cols, "invariant+fixed", False) == ["a_vert__rms", "ax__rms",
                                                             "peak_ratio_h_v"]
    with pytest.raises(ValueError):
        select_columns(cols, "nope", True)
