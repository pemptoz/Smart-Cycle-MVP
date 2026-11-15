import numpy as np
import pandas as pd
import pytest

from smartcycle.features.preprocess import (PipelineConfig, estimate_fs, label_runs,
                                            prepare_gps, preprocess_ride, resample_accel)

G = 9.81


def ride(fs: float, dur: float = 10.0, acc_fn=None, gps: pd.DataFrame | None = None
         ) -> pd.DataFrame:
    n = int(round(dur * fs)) + 1
    t = np.linspace(0.0, dur, n)
    acc = np.column_stack([np.zeros(n), np.zeros(n), np.full(n, G)])
    if acc_fn is not None:
        acc = acc + acc_fn(t)
    df = pd.DataFrame({"t_s": t, "timestamp_ms": np.round(t * 1000).astype("int64"),
                       "accel_x": acc[:, 0], "accel_y": acc[:, 1], "accel_z": acc[:, 2],
                       "latitude": np.nan, "longitude": np.nan, "gps_accuracy_m": np.nan,
                       "speed_mps": np.nan, "label": pd.array([pd.NA] * n, dtype="string")})
    if gps is not None:
        idx = np.searchsorted(t, gps["t_s"].to_numpy())
        for c in ("latitude", "longitude", "gps_accuracy_m", "speed_mps"):
            if c in gps:
                df.loc[idx, c] = gps[c].to_numpy()
    return df


@pytest.mark.parametrize("fs", [37.0, 100.0, 200.0])
def test_resampling_same_length_for_any_native_rate(fs):
    prep = preprocess_ride(ride(fs), PipelineConfig(fs=50.0))
    assert len(prep.t) == 501
    assert np.allclose(np.diff(prep.t), 0.02)
    assert prep.native_fs == pytest.approx(fs, rel=1e-3)
    assert set(prep.signals) == {"a_vert", "a_hmag", "a_dev", "ax", "ay", "az"}


def test_estimate_fs_robust_to_millisecond_rounding_and_gaps():
    t = np.round(np.arange(0, 20, 1 / 80.0) * 1000) / 1000  # 80 Hz rounded to whole ms
    t = np.concatenate([t[:800], t[900:]])  # a 1.25 s gap
    assert estimate_fs(t) == pytest.approx(80.0, rel=0.01)


def test_anti_alias_when_downsampling():
    fs_in = 200.0
    t = np.arange(0, 10, 1 / fs_in)
    x = np.column_stack([np.sin(2 * np.pi * 40 * t)] * 3)  # 40 Hz > 25 Hz Nyquist of 50 Hz
    _, out, valid, native = resample_accel(t, x, 50.0)
    assert native == pytest.approx(200.0)
    assert np.abs(out[100:-100]).max() < 0.1  # aliased tone strongly attenuated
    assert valid.all()


def test_gap_marks_samples_invalid():
    t = np.concatenate([np.arange(0, 5, 0.01), np.arange(6, 10, 0.01)])
    _, _, valid, _ = resample_accel(t, np.zeros((len(t), 3)), 50.0)
    assert not valid.all() and valid.sum() > 0.85 * len(valid)


def test_vertical_bump_goes_to_a_vert_not_horizontal():
    def bump(t):
        out = np.zeros((len(t), 3))
        out[:, 2] = 8.0 * np.exp(-((t - 5.0) / 0.05) ** 2)
        return out
    prep = preprocess_ride(ride(100.0, acc_fn=bump), PipelineConfig())
    k = np.argmax(np.abs(prep.signals["a_vert"]))
    assert abs(prep.t[k] - 5.0) < 0.05
    assert prep.signals["a_vert"][k] > 6.0
    assert prep.signals["a_hmag"].max() < 1.0


def test_gps_accuracy_filter_and_haversine_speed():
    t = np.arange(0.5, 10, 1.0)
    lat = 53.35 + (5.0 * t) / 111_195.0  # 5 m/s due north
    acc = np.where(np.arange(len(t)) == 3, 80.0, 5.0)
    gps = pd.DataFrame({"t_s": t, "latitude": lat, "longitude": -6.26, "gps_accuracy_m": acc})
    fixes, info = prepare_gps(ride(50.0, gps=gps), max_accuracy_m=30.0)
    assert info["n_fixes_dropped_accuracy"] == 1 and len(fixes) == len(t) - 1
    assert info["speed_source"].startswith("haversine")
    assert np.nanmedian(fixes["speed_mps"]) == pytest.approx(5.0, rel=0.05)
    prep = preprocess_ride(ride(50.0, gps=gps), PipelineConfig())
    assert prep.has_speed and np.nanmean(prep.speed) == pytest.approx(5.0, rel=0.05)


def test_no_gps_means_no_speed():
    prep = preprocess_ride(ride(50.0), PipelineConfig())
    assert not prep.has_speed and np.isnan(prep.speed).all()


def test_label_runs():
    t = np.arange(10) * 0.5
    lab = pd.Series(["Smooth"] * 4 + [None] * 2 + ["Rough"] * 4, dtype="string")
    assert label_runs(t, lab) == [(0.0, 2.0, "Smooth"), (3.0, 5.0, "Rough")]
