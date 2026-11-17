"""Network segmentation, OSM parsing, map matching, severity (no network access)."""

import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from smartcycle.cli import main
from smartcycle.geo import osm
from smartcycle.geo.matching import match_stats, match_track, snap_point
from smartcycle.geo.network import (Projector, network_from_geojson, polyline_length_m,
                                    segmentise)
from smartcycle.geo.severity import band, segment_score, severity_from_value
from smartcycle.synth.routes import ROUTES, Polyline, routes_geojson

LAT0, LON0 = 53.345, -6.26
P = Projector(LAT0, LON0)


def ll(x, y):
    """metres -> [lon, lat]"""
    la, lo = P.to_latlon(np.asarray(x, float), np.asarray(y, float))
    return np.column_stack([lo, la]).tolist()


def line(name, pts, **props):
    return {"type": "Feature", "properties": {"route_id": name, **props},
            "geometry": {"type": "LineString", "coordinates": ll(*np.array(pts).T)}}


def fixes_from_xy(x, y, dt=1.0, acc=6.0, speed=5.0):
    la, lo = P.to_latlon(np.asarray(x, float), np.asarray(y, float))
    return pd.DataFrame({"t_s": np.arange(len(la)) * dt, "latitude": la, "longitude": lo,
                         "gps_accuracy_m": acc, "speed_mps": speed})


def test_segmentise_lengths_and_stable_ids():
    pieces = segmentise(ll([0, 430], [0, 0]))
    lens = [polyline_length_m(p) for p in pieces]
    assert len(pieces) == 5 and all(abs(v - 86.0) < 0.5 for v in lens)
    assert len(segmentise(ll([0, 60], [0, 0]))) == 1
    net = network_from_geojson(routes_geojson())
    net2 = network_from_geojson(routes_geojson())
    assert net.ids == net2.ids and net.ids[0] == "seg_syn_quays_0"
    assert all(50 <= p["length_m"] <= 100.5 for p in net.props if p["way_id"] != "x")
    i0, i1 = net.index["seg_syn_quays_0"], net.index["seg_syn_quays_1"]
    assert net.connected(i0, i1) and net.synthetic


def test_noisy_track_recovers_true_segments():
    net = network_from_geojson(routes_geojson())
    rng = np.random.default_rng(3)
    for route in ("syn_diagonal", "syn_canal"):
        pl = Polyline(ROUTES[route])
        s = np.arange(0, pl.length * 1.5, 5.0)
        pts = pl.at(s)
        truth = [snap_point(a, b, net)["segment_id"] for a, b in pts]
        e = np.zeros((len(s), 2))
        for i in range(1, len(s)):  # AR(1) noise, ~3-6 m like the synthetic GPS
            e[i] = 0.8 * e[i - 1] + rng.normal(0, 3.0, 2)
        k = np.cos(np.radians(pts[:, 0]))
        df = pd.DataFrame({"t_s": s / 5.0, "latitude": pts[:, 0] + e[:, 1] / 111_320,
                           "longitude": pts[:, 1] + e[:, 0] / (111_320 * k),
                           "gps_accuracy_m": 6.0, "speed_mps": 5.0})
        m = match_track(df, net)
        acc = (m["segment_id"].to_numpy() == np.array(truth, dtype=object)).mean()
        assert acc >= 0.90, (route, acc)
        assert match_stats(m)["pct_matched"] == 100.0


def test_far_track_is_unmatched():
    net = network_from_geojson(routes_geojson())
    df = pd.DataFrame({"t_s": [0.0, 1.0, 2.0], "latitude": [53.40] * 3,
                       "longitude": [-6.20, -6.2001, -6.2002], "gps_accuracy_m": 5.0})
    m = match_track(df, net)
    assert not m["matched"].any() and m["segment_id"].isna().all()
    assert m["snapped_lat"].tolist() == df["latitude"].tolist()  # raw GPS kept
    assert not snap_point(53.40, -6.20, net)["matched"]
    assert not match_track(df, None)["matched"].any()


def test_connectivity_beats_a_closer_unconnected_segment():
    gj = {"type": "FeatureCollection", "features": [
        line("A", [(0, 0), (200, 0)]),         # -> seg_A_0, seg_A_1 (connected)
        line("B", [(80, 12), (130, 12)]),      # parallel, NOT connected
    ]}
    net = network_from_geojson(gj)
    x = np.arange(0, 200, 5.0)
    y = np.full_like(x, 1.0)
    y[(x >= 100) & (x <= 110)] = 9.0  # 3 fixes much closer to B than to A
    m = match_track(fixes_from_xy(x, y), net)
    assert set(m["segment_id"]) == {"seg_A_0", "seg_A_1"}
    near_b = snap_point(*P.to_latlon(105.0, 9.0), net)
    assert near_b["segment_id"] == "seg_B_0"  # nearest alone would pick B


def test_fallback_nearest_within_40m():
    gj = {"type": "FeatureCollection", "features": [line("A", [(0, 0), (90, 0)])]}
    net = network_from_geojson(gj)
    m = match_track(fixes_from_xy([10, 20, 30], [30, 30, 30], acc=3.0), net)  # r = 20 m
    assert m["matched"].all() and (m["snap_dist_m"] - 30).abs().max() < 0.5


FAKE_OVERPASS = {"elements": [
    {"type": "way", "id": 1, "nodes": [10, 11, 12], "tags": {"highway": "cycleway",
                                                               "name": "Canal Way"}},
    {"type": "way", "id": 2, "nodes": [11, 13], "tags": {"highway": "residential",
                                                         "cycleway": "separate"}},
    {"type": "way", "id": 3, "nodes": [12, 14], "tags": {"highway": "footway"}},
    {"type": "way", "id": 4, "nodes": [13, 14], "tags": {"highway": "tertiary",
                                                         "cycleway:left": "lane"}},
    {"type": "node", "id": 10, "lat": 53.3400, "lon": -6.2600},
    {"type": "node", "id": 11, "lat": 53.3400, "lon": -6.2585},
    {"type": "node", "id": 12, "lat": 53.3400, "lon": -6.2570},
    {"type": "node", "id": 13, "lat": 53.3410, "lon": -6.2585},
    {"type": "node", "id": 14, "lat": 53.3400, "lon": -6.2560},
]}


def test_osm_tag_classification():
    assert osm.is_cycleway({"highway": "cycleway"})
    assert osm.is_cycleway({"highway": "path", "bicycle": "designated"})
    assert not osm.is_cycleway({"highway": "footway"})
    assert osm.is_cycleway({"highway": "secondary", "cycleway:right": "track"})
    assert not osm.is_cycleway({"highway": "residential", "cycleway": "no"})
    assert not osm.is_cycleway({"highway": "residential", "cycleway": "separate"})
    assert osm.keep_way({"highway": "primary"}) and osm.keep_way({"highway": "service"})
    # P05: plain paths / footways are kept as ordinary (is_cycleway=false) ways, not sidewalks
    assert osm.keep_way({"highway": "footway"}) and osm.keep_way({"highway": "path"})
    assert not osm.keep_way({"highway": "footway", "footway": "sidewalk"})
    assert not osm.keep_way({"highway": "steps"})
    q = osm.overpass_query((53.3, -6.3, 53.4, -6.2))
    assert '"highway"="cycleway"' in q and "primary" in q and "53.300000,-6.300000" in q
    assert "footway" in q and "living_street" in q


def test_overpass_to_segments_splits_at_junctions():
    gj = osm.overpass_to_segments(FAKE_OVERPASS)
    props = {f["properties"]["segment_id"]: f["properties"] for f in gj["features"]}
    assert not props["seg_3_0"]["is_cycleway"]  # P05: plain footway kept, ordinary layer
    way1 = sorted(k for k in props if k.startswith("seg_1_"))
    assert len(way1) == 2  # split at node 11 (shared with way 2), ~100 m each
    assert props["seg_1_0"]["is_cycleway"] and props["seg_1_0"]["name"] == "Canal Way"
    assert not props["seg_2_0"]["is_cycleway"] and props["seg_4_0"]["is_cycleway"]
    assert all(50 <= p["length_m"] <= 100.5 for p in props.values() if p["way_id"] == 1)
    net = network_from_geojson(gj)
    i = net.index
    assert net.connected(i["seg_1_0"], i["seg_1_1"]) and net.connected(i["seg_1_0"], i["seg_2_0"])
    assert not net.connected(i["seg_1_0"], i["seg_4_0"])


def test_fetch_osm_uses_cache_and_fails_cleanly(tmp_path, monkeypatch, capsys):
    out = tmp_path / "net.geojson"
    out.write_text(json.dumps(osm.overpass_to_segments(FAKE_OVERPASS)))
    path, n, downloaded = osm.fetch_osm(tmp_path / "nowhere", out)
    assert not downloaded and n > 0

    def boom(*a, **k):
        raise urllib.error.URLError("offline")
    monkeypatch.setattr(osm.urllib.request, "urlopen", boom)
    monkeypatch.setattr(osm.time, "sleep", lambda s: None)
    with pytest.raises(osm.OverpassError):
        osm.fetch_overpass("x")
    track = tmp_path / "t.csv"
    track.write_text("timestamp_ms,accel_x,accel_y,accel_z,latitude,longitude\n"
                     "0,0,0,9.8,53.34,-6.26\n20,0,0,9.8,,\n40,0,0,9.8,53.3401,-6.2601\n")
    rc = main(["fetch-osm", "--tracks", str(track), "--out", str(tmp_path / "new.geojson")])
    assert rc == 3 and "matched=false" in capsys.readouterr().err
    assert not (tmp_path / "new.geojson").exists()


def test_severity_bins_and_score():
    assert severity_from_value("Pothole", 5.9) == 1
    assert severity_from_value("Pothole", 6.0) == 2
    assert severity_from_value("Pothole", 18.0) == 5
    assert severity_from_value("Irregularity", 4.0) == 2
    assert severity_from_value("Irregularity", 12.5) == 5
    assert severity_from_value("Rough", float("nan")) == 1
    assert band(segment_score(6, 0, 100)) == "hi"
    assert band(segment_score(6, 0, 300)) == "lo"  # 2 points per 100 m
    assert segment_score(0, 75, 50) == 3.0 and band(3.0) == "mid"


def test_accuracy_aware_radius_and_1031m_outlier():
    """r = max(20 m, 2 x accuracy). A fix with a huge accuracy (1031 m, seen on the real irregs
    ride) displaced 300 m off the path must not break the chain of its neighbours: they stay on
    the true segment (the outlier itself has a near-flat emission, sigma = 1031 m)."""
    gj = {"type": "FeatureCollection", "features": [line("A", [(0, 0), (200, 0)]),
                                                    line("B", [(0, 300), (200, 300)])]}
    net = network_from_geojson(gj)
    x = np.arange(0, 200, 20.0)
    y = np.full(len(x), 4.0)
    acc = np.full(len(x), 14.2)
    y[5], acc[5] = 300.0, 1031.0
    m = match_track(fixes_from_xy(x, y, acc=acc, speed=2.0), net)
    assert m["matched"].all()
    others = m.drop(index=5)
    assert others["segment_id"].str.startswith("seg_A").all()
    assert (others["snap_dist_m"] < 5).all()
    # 18 m off a path with 6 m accuracy: inside r = max(20, 12) = 20 m (was 15 m before P05)
    m2 = match_track(fixes_from_xy([10, 20, 30], [18, 18, 18], acc=6.0), net)
    assert m2["matched"].all() and m2["segment_id"].str.startswith("seg_A").all()
