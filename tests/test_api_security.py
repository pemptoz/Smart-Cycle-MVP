"""P90 review regressions: write authorisation (delete token / partner key), public JSON
allow-list, CPU work off the event loop, demo partner key refusal, broken model file.
Temp DB, synthetic network, tiny synthetic model (conftest.py). No network access."""

import asyncio
import dataclasses
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from smartcycle.api import app as app_mod
from smartcycle.api.app import DEFAULT_PARTNER_KEY, Settings, check_partner_key, create_app
from tests.test_api import KEY, make_client, upload

# keys that must never appear anywhere in a public (no partner key) response
FORBIDDEN = {"ride_id", "device_id", "severity", "severity_note", "confidence", "peak",
             "segment_id", "raw_lat", "raw_lon", "source", "started_at", "summary",
             "delete_token_hash", "public_handle", "n_samples", "created_at", "detail",
             "photo_url", "score", "length_m"}


def keys_anywhere(o) -> set[str]:
    out: set[str] = set()
    if isinstance(o, dict):
        for k, v in o.items():
            out.add(k)
            out |= keys_anywhere(v)
    elif isinstance(o, list):
        for v in o:
            out |= keys_anywhere(v)
    return out


@pytest.fixture
def client(tmp_path, tiny_model, routes_file):
    c, s = make_client(tmp_path, tiny_model, routes_file)
    c.settings = s
    return c


@pytest.fixture(scope="session")
def ride_csv(tiny_data):
    return tiny_data / "rides" / "syn_00.csv"


def tok(body: dict) -> dict:
    return {"X-Delete-Token": body["delete_token"]}


# ---------------------------------------------------------------- write authorisation
def test_delete_needs_the_ride_token_or_partner_key(client, ride_csv):
    a = upload(client, ride_csv).json()
    b = upload(client, ride_csv).json()
    assert a["delete_token"] and a["delete_token"] != b["delete_token"]
    url = f"/api/rides/{a['ride']}"
    assert client.delete(url).status_code == 403                                  # missing
    assert client.delete(url, headers={"X-Delete-Token": "x" * 32}).status_code == 403  # wrong
    assert client.delete(url, headers=tok(b)).status_code == 403          # another ride's token
    assert client.delete(url, headers={"X-Partner-Key": "nope"}).status_code == 403
    assert client.get(url).status_code == 200                              # nothing deleted
    assert client.delete(url, headers=tok(a)).status_code == 200           # correct token
    assert client.get(url).status_code == 404
    assert client.delete(f"/api/rides/{b['ride']}", headers=KEY).status_code == 200  # partner


def test_detail_and_photo_need_a_token_of_a_reporting_ride(client, ride_csv):
    a = upload(client, ride_csv).json()
    b = upload(client, ride_csv).json()
    own = [e["id"] for e in a["events"]]
    eid = own[0]
    other_only = [e["id"] for e in b["events"] if e["id"] not in own]
    png = ("p.png", b"\x89PNG\r\n\x1a\n" + b"x" * 50, "image/png")
    body = {"reason": "glass", "text": "bottle"}
    assert client.post(f"/api/events/{eid}/detail", json=body).status_code == 403
    assert client.post(f"/api/events/{eid}/detail", json=body,
                       headers={"X-Delete-Token": "wrong"}).status_code == 403
    assert client.post(f"/api/events/{eid}/photo", files={"file": png}).status_code == 403
    assert client.post(f"/api/events/{eid}/detail", json=body, headers=tok(a)).status_code == 200
    assert client.post(f"/api/events/{eid}/detail", json=body, headers=KEY).status_code == 200
    # a token of ride A cannot edit an event reported only by ride B
    if other_only:
        assert client.post(f"/api/events/{other_only[0]}/detail", json=body,
                           headers=tok(a)).status_code == 403
    # merged events: either reporting ride's token works
    merged = [e["id"] for e in b["events"] if e["merged_into_existing"]]
    if merged:
        assert client.post(f"/api/events/{merged[0]}/detail", json=body,
                           headers=tok(b)).status_code == 200
    # unknown event -> 404 (before any auth decision leaks nothing: ids are public anyway)
    assert client.post("/api/events/A-9999/detail", json=body, headers=tok(a)).status_code == 404


def test_cli_ingested_ride_has_no_token_only_partner_can_write(client, ride_csv, tiny_model):
    """Rides stored by `ingest` (no delete_token_hash): no token can match, the partner key can."""
    from contextlib import closing

    import joblib

    from smartcycle.api import db
    from smartcycle.geo.network import load_network
    from smartcycle.io.loader import load_ride
    from smartcycle.pipeline import process_ride

    res = process_ride(load_ride(ride_csv), joblib.load(tiny_model[0]),
                       load_network(client.settings.network_path))
    with closing(db.connect(client.settings.db_path)) as c:
        rid, evs = db.store_ride(c, res, device_id="anon-cli", source="cli:x.csv",
                                 provenance="synthetic", mount="handlebar")
        handle = db.get_ride(c, rid)["public_handle"]
    other = upload(client, ride_csv).json()  # a valid token, for another ride
    assert client.delete(f"/api/rides/{handle}", headers=tok(other)).status_code == 403
    assert client.delete(f"/api/rides/{handle}",
                         headers={"X-Delete-Token": ""}).status_code == 403
    assert client.post(f"/api/events/{evs[0]['id']}/detail",
                       json={"reason": "other"}).status_code == 403
    assert client.delete(f"/api/rides/{handle}", headers=KEY).status_code == 200


def test_readonly_refuses_writes_even_with_valid_token(tmp_path, tiny_model, routes_file,
                                                       ride_csv):
    rw, s = make_client(tmp_path, tiny_model, routes_file)
    up = upload(rw, ride_csv).json()
    ro = TestClient(create_app(dataclasses.replace(s, readonly=True)))
    eid = up["events"][0]["id"]
    for r in (ro.delete(f"/api/rides/{up['ride']}", headers=tok(up)),
              ro.post(f"/api/events/{eid}/detail", json={"reason": "other"}, headers=tok(up)),
              ro.delete(f"/api/rides/{up['ride']}", headers=KEY)):
        assert r.status_code == 403 and r.json()["readonly"] is True
    assert ro.get(f"/api/rides/{up['ride']}").status_code == 200


# ---------------------------------------------------------------- public JSON allow-list
def test_public_json_allow_list(client, ride_csv):
    up = upload(client, ride_csv)
    body = up.json()
    assert not keys_anywhere({k: v for k, v in body.items() if k != "delete_token"}) & FORBIDDEN
    handle = body["ride"]
    assert len(handle) == 16 and int(handle, 16) >= 0  # random 16-hex public handle
    pages = {p: client.get(p).json() for p in
             ("/api/rides", f"/api/rides/{handle}", "/api/events", "/api/health")}
    # /api/health network.source is the OSM / synthetic network file, not a ride
    pages["/api/health"] = {k: v for k, v in pages["/api/health"].items() if k != "network"}
    for path, j in pages.items():
        leaked = keys_anywhere(j) & FORBIDDEN
        assert not leaked, (path, leaked)
    assert "id" not in pages["/api/rides"]["rides"][0]  # no ride_id under another name
    r = pages[f"/api/rides/{handle}"]
    assert set(r) == {"handle", "date", "date_kind", "provenance", "mount", "duration_s",
                      "distance_m", "n_events", "track", "events", "note"}
    assert len(r["date"]) == 10  # day only, never the start time
    assert all(set(e) == set(app_mod.PUBLIC_RIDE_EVENT_FIELDS) for e in r["events"])
    # track: only map-matched (snapped) points of the partner track
    part = client.get(f"/api/rides/{handle}", headers=KEY).json()
    assert part["id"] and part["source"].startswith("csv:") and "summary" in part
    pt = part["track"]
    snapped = {tuple(c) for c, m in zip(pt["geometry"]["coordinates"],
                                        pt["properties"]["matched"]) if m}
    assert {tuple(c) for c in r["track"]["geometry"]["coordinates"]} <= snapped
    assert len(r["track"]["properties"]["t_ms"]) == len(r["track"]["geometry"]["coordinates"])
    # the model card / metrics are public by design but never contain ride identifiers
    # the model card / metrics and the OSM network (segment_id, length_m of the base map) are
    # public by design but never contain ride identifiers
    for path in ("/api/model", "/api/network"):
        assert not keys_anywhere(client.get(path).json()) & {"ride_id", "device_id", "source",
                                                              "started_at", "public_handle"}


def test_public_track_drops_unmatched_raw_gps(tmp_path, tiny_model, ride_csv):
    c, _ = make_client(tmp_path, tiny_model, None)  # no network: every fix is raw GPS
    up = upload(c, ride_csv).json()
    pub = c.get(f"/api/rides/{up['ride']}").json()
    assert pub["track"]["geometry"]["coordinates"] == []
    part = c.get(f"/api/rides/{up['ride']}", headers=KEY).json()
    assert len(part["track"]["geometry"]["coordinates"]) > 10


def test_ride_handles_are_validated(client):
    for bad in ("A" * 16, "abc", "0" * 17, "' OR 1=1 --"):
        assert client.get(f"/api/rides/{bad}").status_code == 404
        assert client.delete(f"/api/rides/{bad}", headers=KEY).status_code == 404
    assert client.get("/api/rides/../t.db").status_code == 404  # normalised away, not served


def test_db_migration_adds_handles_to_old_rides(tmp_path):
    import sqlite3
    from contextlib import closing

    from smartcycle.api import db

    p = tmp_path / "old.db"
    with closing(sqlite3.connect(p)) as c:  # the rides table as committed before P90
        c.execute("CREATE TABLE rides (id TEXT PRIMARY KEY, device_id TEXT NOT NULL, started_at"
                  " TEXT, source TEXT, mount TEXT, provenance TEXT NOT NULL, n_samples INTEGER,"
                  " duration_s REAL, distance_m REAL, summary TEXT, created_at TEXT NOT NULL)")
        c.execute("INSERT INTO rides (id, device_id, provenance, created_at) VALUES"
                  " ('r1', 'anon-1', 'real', '2026-11-16T17:00:00+00:00')")
        c.commit()
    for _ in range(2):  # idempotent
        with closing(db.connect(p)) as c:
            row = dict(c.execute("SELECT * FROM rides").fetchone())
    assert len(row["public_handle"]) == 16 and row["delete_token_hash"] is None


# ---------------------------------------------------------------- event loop
def test_csv_processing_does_not_block_the_event_loop(tmp_path, tiny_model, routes_file,
                                                      ride_csv, monkeypatch):
    real = app_mod.process_ride

    def slow(*a, **k):
        time.sleep(1.5)  # CPU-bound stand-in: blocks its thread
        return real(*a, **k)

    monkeypatch.setattr(app_mod, "process_ride", slow)
    _, s = make_client(tmp_path, tiny_model, routes_file)
    app = create_app(s)
    data = ride_csv.read_bytes()

    async def run():
        tr = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=tr, base_url="http://t") as ac:
            t0 = time.perf_counter()
            up = asyncio.create_task(ac.post("/api/rides/upload-csv",
                                             files={"file": ("r.csv", data, "text/csv")}))
            await asyncio.sleep(0.3)
            h = await ac.get("/api/health")
            t_health = time.perf_counter() - t0
            r = await up
            return h.status_code, t_health, r.status_code, time.perf_counter() - t0

    hs, t_health, us, t_up = asyncio.run(run())
    assert hs == 200 and us == 200
    assert t_health < 1.0 < t_up, (t_health, t_up)  # health answered while the upload ran


# ---------------------------------------------------------------- partner key
def test_writable_server_refuses_the_demo_key(tmp_path, monkeypatch):
    base = Settings(db_path=tmp_path / "t.db", model_path=tmp_path / "m.joblib",
                    network_path=tmp_path / "n.geojson", photos_dir=tmp_path / "p",
                    web_dir=tmp_path, card_path=tmp_path / "c.json",
                    metrics_path=tmp_path / "x.json")
    assert base.partner_key == DEFAULT_PARTNER_KEY
    with pytest.raises(RuntimeError, match="PARTNER_KEY"):
        check_partner_key(base)
    check_partner_key(dataclasses.replace(base, readonly=True))
    check_partner_key(dataclasses.replace(base, allow_demo_key=True))
    check_partner_key(dataclasses.replace(base, partner_key="s3cret"))
    monkeypatch.setenv("SMARTCYCLE_ALLOW_DEMO_KEY", "1")
    assert Settings.from_env().allow_demo_key is True
    monkeypatch.delenv("SMARTCYCLE_ALLOW_DEMO_KEY")
    monkeypatch.setenv("PARTNER_KEY", "")
    assert Settings.from_env().partner_key == DEFAULT_PARTNER_KEY  # empty env = default


def test_serve_cli_refuses_demo_key(monkeypatch, capsys):
    from smartcycle.cli import main

    for v in ("SMARTCYCLE_READONLY", "SMARTCYCLE_ALLOW_DEMO_KEY", "PARTNER_KEY"):
        monkeypatch.delenv(v, raising=False)
    assert main(["serve", "--port", "1"]) == 1
    assert "default partner key" in capsys.readouterr().err


def test_partner_key_hint_only_in_readonly_with_default_key(tmp_path, tiny_model, routes_file):
    _, s = make_client(tmp_path, tiny_model, routes_file)
    demo = dataclasses.replace(s, partner_key=DEFAULT_PARTNER_KEY)
    get = lambda st: TestClient(create_app(st)).get("/api/health").json()["partner_key_hint"]  # noqa: E731
    assert get(dataclasses.replace(demo, readonly=True)) == "demo-partner"
    assert get(demo) is None
    assert get(dataclasses.replace(s, readonly=True)) is None  # custom key never shown


def test_dashboard_never_prefills_or_autosends_the_key():
    from tests.test_api import WEB
    html = (WEB / "index.html").read_text()
    js = (WEB / "assets" / "dashboard.js").read_text()
    assert 'value="demo-partner"' not in html
    start = js[js.index("-- start\n"):]
    assert "DEFAULT_KEY" not in start and "demo-partner" not in start
    assert "partner_key_hint" in js and "X-Delete-Token" in js and "sessionStorage" in js


# ---------------------------------------------------------------- broken model file
@pytest.mark.parametrize("kind", ["corrupt", "version"])
def test_broken_model_never_crashes_the_api(tmp_path, tiny_model, routes_file, ride_csv, kind):
    mpath, cpath = tiny_model
    d = tmp_path / "m"
    d.mkdir()
    m2, c2 = d / "model.joblib", d / "model_card.json"
    card = json.loads(cpath.read_text())
    if kind == "corrupt":
        m2.write_bytes(b"not a pickle at all")
    else:
        m2.write_bytes(mpath.read_bytes())
        card["sklearn_version"] = "0.0.1"
    c2.write_text(json.dumps(card))
    c, _ = make_client(tmp_path, (m2, c2), routes_file)
    h = c.get("/api/health")
    assert h.status_code == 200 and h.json()["model_loaded"] is False
    assert "Reproduce" in h.json()["hint"] and h.json()["model_load_error"]
    m = c.get("/api/model").json()
    assert m["loaded"] is False and "Reproduce" in m["hint"]
    r = upload(c, ride_csv)
    assert r.status_code == 503 and "Reproduce" in r.json()["error"]
