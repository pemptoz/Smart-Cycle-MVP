"""FastAPI happy path, public/partner views, dedup, photos (EXIF stripping), deletion, fallbacks.
Temp DB, synthetic network, tiny synthetic model (see conftest.py). No network access."""

import datetime as dt
import io
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from smartcycle.api import db
from smartcycle.api.app import Settings, create_app
from smartcycle.io.loader import load_ride

KEY = {"X-Partner-Key": "test-key"}
WEB = Path(__file__).resolve().parents[1] / "web"


def make_client(tmp_path, tiny_model, network: Path | None, model=True):
    mpath, cpath = tiny_model
    s = Settings(db_path=tmp_path / "t.db",
                 model_path=mpath if model else tmp_path / "none.joblib",
                 network_path=network or tmp_path / "missing.geojson",
                 photos_dir=tmp_path / "photos", web_dir=WEB, card_path=cpath,
                 metrics_path=tmp_path / "none.json", partner_key="test-key")
    return TestClient(create_app(s)), s


@pytest.fixture
def client(tmp_path, tiny_model, routes_file):
    c, s = make_client(tmp_path, tiny_model, routes_file)
    c.settings = s
    return c


@pytest.fixture(scope="session")
def ride_csv(tiny_data):
    return tiny_data / "rides" / "syn_00.csv"


def upload(client, path, headers=None, **form):
    with open(path, "rb") as f:
        return client.post("/api/rides/upload-csv", files={"file": (path.name, f, "text/csv")},
                           data={"provenance": "synthetic", **form}, headers=headers or {})


def jpeg_with_gps_exif() -> bytes:
    im = Image.new("RGB", (2000, 1500), (120, 160, 90))
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[2] = (53.0, 20.0, 10.0)
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    data = buf.getvalue()
    assert Image.open(io.BytesIO(data)).getexif().get_ifd(0x8825)  # really has GPS EXIF
    return data


def test_health_and_root(client):
    h = client.get("/api/health").json()
    assert h["model_loaded"] and h["model_provenance"] == "synthetic"
    assert h["osm_loaded"] and h["network"]["synthetic"]
    assert client.get("/api/model").json()["model_card"]["data_provenance"] == "synthetic"
    assert client.get("/").status_code == 200


def test_upload_happy_path_public_hides_severity(client, ride_csv):
    r = upload(client, ride_csv)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["matched"] and body["summary"]["gps"]["pct_matched"] > 90
    assert len(body["events"]) > 0
    assert all("severity" not in e for e in body["events"])
    pub = client.get("/api/events").json()
    assert pub["view"] == "public" and len(pub["features"]) > 0
    for f in pub["features"]:
        assert set(f["properties"]) == {"id", "type", "lat", "lon", "date", "reports",
                                        "provenance"}
    assert pub["data_provenance"] == "synthetic" and pub["model_provenance"] == "synthetic"
    ride = client.get(f"/api/rides/{body['ride_id']}").json()
    assert len(ride["track"]["geometry"]["coordinates"]) > 10
    assert all("t_peak_ms" in e and "severity" not in e for e in ride["events"])


def test_partner_view_requires_key(client, ride_csv):
    upload(client, ride_csv)
    assert client.get("/api/events?view=partner").status_code == 401
    assert client.get("/api/events?view=partner",
                      headers={"X-Partner-Key": "nope"}).status_code == 401
    for path in ("/api/partner/segments", "/api/partner/alerts", "/api/partner/priorities"):
        assert client.get(path).status_code == 401
    part = client.get("/api/events?view=partner", headers=KEY).json()
    p = part["features"][0]["properties"]
    assert 1 <= p["severity"] <= 5 and "uncalibrated" in p["severity_note"]
    assert p["segment_id"].startswith("seg_syn_") and p["matched"]
    seg = client.get("/api/partner/segments", headers=KEY).json()
    assert seg["features"] and {"score", "band", "covered_m"} <= set(
        seg["features"][0]["properties"])
    pri = client.get("/api/partner/priorities", headers=KEY).json()["priorities"]
    assert 0 < len(pri) <= 5 and pri[0]["score"] >= pri[-1]["score"]
    al = client.get("/api/partner/alerts", headers=KEY).json()
    assert al["reference"] in ("wall_clock", "latest_event") and len(al["alerts"]) <= 4
    assert all(a["severity"] >= 4 for a in al["alerts"])
    bb = client.get("/api/events?bbox=0,0,1,1").json()
    assert bb["features"] == []
    assert client.get("/api/events?bbox=abc").status_code == 400
    assert client.get("/api/events?type=Crack").status_code == 422


def test_dedup_increments_reports(client, ride_csv):
    first = upload(client, ride_csv).json()
    second = upload(client, ride_csv).json()
    merged = [e for e in second["events"] if e["merged_into_existing"]]
    assert merged and all(e["type"] in ("Pothole", "Irregularity") for e in merged)
    ev = {f["properties"]["id"]: f["properties"] for f in client.get("/api/events").json()
          ["features"]}
    assert all(ev[e["id"]]["reports"] == 2 for e in merged)
    n_point_first = sum(e["type"] != "Rough" for e in first["events"])
    assert sum(p["type"] != "Rough" for p in ev.values()) == n_point_first


def test_json_batch_from_phone(client, ride_csv):
    df = load_ride(ride_csv)
    t = (1_790_000_000_000 + df["timestamp_ms"]).astype(float)
    samples = [{"t": a, "ax": x, "ay": y, "az": z} for a, x, y, z in
               zip(t, df["accel_x"], df["accel_y"], df["accel_z"])]
    g = df[df["latitude"].notna()]
    gps = [{"t": a, "lat": la, "lon": lo, "acc": 5.0,
            "speed": None if s != s else s} for a, la, lo, s in
           zip(t[g.index], g["latitude"], g["longitude"], g["speed_mps"])]
    r = client.post("/api/rides", json={"samples": samples, "gps": gps, "mount": "handlebar",
                                        "device_id": "anon-test", "provenance": "synthetic"},
                    headers=KEY)
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["matched"] and len(b["events"]) > 0 and "severity" in b["events"][0]
    assert b["summary"]["started_at"].startswith("2026-")
    bad = client.post("/api/rides", json={"samples": samples[:20], "device_id": "../etc"})
    assert bad.status_code == 422 and bad.json()["error"] == "invalid request"


def test_photo_exif_stripped_and_delete_removes_files(client, ride_csv):
    ride = upload(client, ride_csv).json()
    eid = ride["events"][0]["id"]
    r = client.post(f"/api/events/{eid}/photo",
                    files={"file": ("p.jpg", jpeg_with_gps_exif(), "image/jpeg")})
    assert r.status_code == 200, r.text
    assert "plates" in r.json()["privacy"] and r.json()["metadata_stripped"]
    url = r.json()["photo_url"]
    img = client.get(url)
    assert img.status_code == 200 and b"Exif" not in img.content
    im = Image.open(io.BytesIO(img.content))
    assert len(im.getexif()) == 0 and max(im.size) == 1280
    files = list(client.settings.photos_dir.iterdir())
    assert len(files) == 1
    part = client.get("/api/events?view=partner", headers=KEY).json()
    assert any(f["properties"]["photo_url"] == url for f in part["features"])

    d = client.delete(f"/api/rides/{ride['ride_id']}")
    assert d.status_code == 200 and d.json()["photos_deleted"] == 1
    assert list(client.settings.photos_dir.iterdir()) == []
    assert client.get(f"/api/rides/{ride['ride_id']}").status_code == 404
    assert client.get("/api/events").json()["features"] == []
    assert client.delete(f"/api/rides/{ride['ride_id']}").status_code == 404


def test_delete_keeps_events_backed_by_another_ride(client, ride_csv):
    a = upload(client, ride_csv).json()
    b = upload(client, ride_csv).json()
    client.delete(f"/api/rides/{a['ride_id']}")
    ev = client.get("/api/events").json()["features"]
    assert ev and all(f["properties"]["reports"] == 1 for f in ev)
    # rides a and b are identical: b's point events merged into a's, its Rough stretches are
    # new rows. After deleting a, exactly b's events remain.
    assert {f["properties"]["id"] for f in ev} == {e["id"] for e in b["events"]}
    with closing(db.connect(client.settings.db_path)) as c:
        owners = {r[0] for r in c.execute("SELECT ride_id FROM events")}
    assert owners == {b["ride_id"]}


def test_rejects_wrong_files_and_path_tricks(client, ride_csv, tmp_path):
    ride = upload(client, ride_csv).json()
    eid = ride["events"][0]["id"]
    for name, data in (("x.gif", b"GIF89a" + b"\0" * 100), ("x.jpg", b"not an image"),
                       ("x.png", b"\x89PNG\r\n\x1a\n" + b"garbage" * 10)):
        r = client.post(f"/api/events/{eid}/photo", files={"file": (name, data, "image/png")})
        assert r.status_code == 415, (name, r.text)
    r = client.post("/api/rides/upload-csv", files={"file": ("ride.exe", b"MZ\0\0", "x/y")})
    assert r.status_code == 415
    r = client.post("/api/rides/upload-csv",
                    files={"file": ("r.csv", b"foo,bar\n1,2\n3,4\n", "text/csv")})
    assert r.status_code == 422 and "foo" in r.json()["error"]
    assert client.post("/api/events/A-9999/photo", files={
        "file": ("p.jpg", jpeg_with_gps_exif(), "image/jpeg")}).status_code == 404
    for bad in ("..%2F..%2Fetc%2Fpasswd", "abc.jpg", "../t.db"):
        assert client.get(f"/api/photos/{bad}").status_code == 404
    assert client.post(f"/api/events/{eid}/detail",
                       json={"reason": "glass", "text": "broken bottle"}).status_code == 200
    assert client.post(f"/api/events/{eid}/detail", json={"reason": "crack"}).status_code == 422
    assert client.post(f"/api/events/{eid}/detail",
                       json={"reason": "other", "text": "x" * 201}).status_code == 422
    p = client.get("/api/events?view=partner", headers=KEY).json()["features"]
    assert any(f["properties"]["detail"]["reason"] == "glass" for f in p)


def test_unmatched_fallback_without_network(tmp_path, tiny_model, ride_csv):
    c, _ = make_client(tmp_path, tiny_model, None)
    assert c.get("/api/health").json()["osm_loaded"] is False
    r = upload(c, ride_csv, headers=KEY)
    assert r.status_code == 200, r.text
    b = r.json()
    assert not b["matched"] and len(b["events"]) > 0
    assert all(e["segment_id"] is None and not e["matched"] for e in b["events"])
    assert c.get("/api/partner/segments", headers=KEY).json()["features"] == []


def test_no_model_gives_503_with_hint(tmp_path, tiny_model, routes_file, ride_csv):
    c, _ = make_client(tmp_path, tiny_model, routes_file, model=False)
    r = upload(c, ride_csv)
    assert r.status_code == 503 and "train" in r.json()["error"]
    assert c.get("/api/health").json()["model_loaded"] is False


def test_alerts_use_latest_event_when_data_is_old(client, ride_csv):
    upload(client, ride_csv, headers=KEY)
    with closing(db.connect(client.settings.db_path)) as c:
        with c:
            c.execute("UPDATE events SET observed_at = '2020-01-05T10:00:00+00:00',"
                      " severity = 5 WHERE seq = 1")
            c.execute("UPDATE events SET observed_at = '2020-01-01T10:00:00+00:00'"
                      " WHERE seq > 1")
        out = db.alerts(c, now=dt.datetime(2026, 11, 15, tzinfo=dt.timezone.utc))
    assert out["reference"] == "latest_event" and out["reference_time"].startswith("2020-01-05")
    assert out["alerts"][0]["severity"] == 5 and "latest event" in out["note"]


def test_mixed_model_provenance_is_reported(tmp_path, tiny_model, routes_file, ride_csv):
    """A model trained on real + augmented/synthetic windows is `mixed` everywhere."""
    import json

    import joblib

    mpath, cpath = tiny_model
    b = joblib.load(mpath)
    b["data_provenance"] = "mixed"
    (tmp_path / "m").mkdir()
    m2, c2 = tmp_path / "m" / "model.joblib", tmp_path / "m" / "model_card.json"
    joblib.dump(b, m2)
    c2.write_text(json.dumps({"model": "rf", "data_provenance": "mixed", "git_dirty": False,
                              "n_synthetic_windows": 0, "augmentation": {"amplitude": [0.85, 1.15]},
                              "cv_protocol": {"mode": "blocked"}}))
    c, _ = make_client(tmp_path, (m2, c2), routes_file)
    assert c.get("/api/health").json()["model_provenance"] == "mixed"
    card = c.get("/api/model").json()["model_card"]
    assert card["data_provenance"] == "mixed" and card["cv_protocol"]["mode"] == "blocked"
    assert upload(c, ride_csv, provenance="real").status_code == 200
    ev = c.get("/api/events").json()
    assert ev["model_provenance"] == "mixed" and ev["data_provenance"] == "real"
