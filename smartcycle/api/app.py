"""FastAPI service: ride ingestion (CSV upload or phone JSON batch), events (public / partner
views), partner analytics, rider details and photos. Serves `web/` on the same origin.

Partner access is a DEMO-GRADE control: header `X-Partner-Key` compared with env `PARTNER_KEY`
(default `demo-partner`). It is not real authentication.
"""

from __future__ import annotations

import hmac
import json
import os
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from smartcycle.api import db
from smartcycle.api.photos import (MAX_PHOTO_BYTES, PRIVACY_REMINDER, PhotoError, photo_path,
                                   save_photo)
from smartcycle.geo.network import Network, default_network_path, load_network
from smartcycle.geo.severity import SEVERITY_NOTE
from smartcycle.io.csvsniff import decode_bytes, read_csv_any
from smartcycle.io.loader import canonicalise
from smartcycle.pipeline import process_ride

MAX_CSV_BYTES = 20 * 1024 * 1024
MAX_JSON_BYTES = 40 * 1024 * 1024
DEVICE_RE = r"^[A-Za-z0-9_-]{1,64}$"
TYPES = ("Smooth", "Rough", "Pothole", "Irregularity")
NO_MODEL_HINT = ("No trained model found. Run `python -m smartcycle train --data-dir <dir>` "
                 "(e.g. data/synthetic) and retry.")
REPORT_IMAGES = ("confusion_matrix.png", "feature_importance.png")  # shown in the Model dialog
PUBLIC_NOTE = "Public view: reported locations only. Severity and analysis are for partner cities."


# ---------------------------------------------------------------- request models
class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    t: float  # epoch ms
    ax: float
    ay: float
    az: float


class Fix(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    t: float  # epoch ms
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    acc: float | None = Field(default=None, ge=0)
    speed: float | None = Field(default=None, ge=0)


class RideBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[Sample] = Field(min_length=10, max_length=400_000)
    gps: list[Fix] = Field(default_factory=list, max_length=20_000)
    mount: Literal["handlebar", "pocket"] = "handlebar"
    device_id: str | None = Field(default=None, pattern=DEVICE_RE)
    provenance: Literal["real", "synthetic", "unknown"] = "unknown"


class Detail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: Literal["glass", "parked_vehicle", "drain_cover", "pothole_avoided", "other"]
    text: str | None = Field(default=None, max_length=200)


# ---------------------------------------------------------------- state
@dataclass
class Settings:
    db_path: Path
    model_path: Path
    network_path: Path
    photos_dir: Path
    web_dir: Path
    card_path: Path
    metrics_path: Path
    partner_key: str = "demo-partner"
    reports_dir: Path | None = None  # PNGs served read-only at /reports/<name>.png
    notes_path: Path | None = None  # hand-written model notes / caveats (models/model_notes.json)

    @classmethod
    def from_env(cls) -> "Settings":
        model = Path(os.environ.get("SMARTCYCLE_MODEL", "models/model.joblib"))
        return cls(db_path=db.default_db_path(), model_path=model,
                   network_path=default_network_path(),
                   photos_dir=Path(os.environ.get("SMARTCYCLE_PHOTOS", "data/photos")),
                   web_dir=Path(os.environ.get("SMARTCYCLE_WEB", "web")),
                   card_path=model.parent / "model_card.json",
                   metrics_path=Path("reports/metrics.json"),
                   partner_key=os.environ.get("PARTNER_KEY", "demo-partner"),
                   reports_dir=Path(os.environ.get("SMARTCYCLE_REPORTS", "reports")),
                   notes_path=model.parent / "model_notes.json")


@dataclass
class State:
    s: Settings
    bundle: dict | None = None
    net: Network | None = None
    extra: dict = field(default_factory=dict)

    def model(self) -> dict | None:
        """Load lazily, so a model trained after the server started is picked up."""
        if self.bundle is None and self.s.model_path.exists():
            import joblib
            self.bundle = joblib.load(self.s.model_path)
        return self.bundle


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    st = State(s=s, net=load_network(s.network_path))
    app = FastAPI(title="Smart Cycle API", version="0.1.0",
                  description="Road-quality events from smartphone rides. Severity is an "
                              "uncalibrated heuristic. Partner key = demo-only control.")
    app.state.st = st

    def conn():
        return closing(db.connect(s.db_path))

    def is_partner(key: str | None) -> bool:
        return bool(key) and hmac.compare_digest(key.encode(), s.partner_key.encode())

    def require_partner(key: str | None) -> None:
        if not is_partner(key):
            raise HTTPException(401, "partner view requires a valid X-Partner-Key header "
                                     "(demo-only control)")

    def model_or_503() -> dict:
        b = st.model()
        if b is None:
            raise HTTPException(503, NO_MODEL_HINT)
        return b

    # ---- errors as JSON
    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _val(_: Request, exc: RequestValidationError):
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse({"error": "invalid request", "details": details[:20]},
                            status_code=422)

    @app.middleware("http")
    async def _limit_body(request: Request, call_next):
        cl = request.headers.get("content-length")
        if request.method == "POST" and cl and cl.isdigit() and int(cl) > MAX_JSON_BYTES:
            return JSONResponse({"error": "request body too large"}, status_code=413)
        return await call_next(request)

    # ---- helpers
    def provenance_block() -> dict:
        b = st.model()
        with conn() as c:
            data = db.provenance_summary(c)
            values = db.provenance_values(c)
        return {"model_provenance": b.get("data_provenance") if b else None,
                "data_provenance": data, "data_provenance_values": values}

    def event_out(e: dict, partner: bool, with_times: bool = False) -> dict:
        lat = e["snapped_lat"] if e.get("snapped_lat") is not None else e["lat"]
        lon = e["snapped_lon"] if e.get("snapped_lon") is not None else e["lon"]
        p = {"id": e["id"], "type": e["type"], "lat": lat, "lon": lon, "date": e["date"],
             "reports": e["reports"], "provenance": e.get("provenance"),
             "way_name": e.get("way_name")}
        if with_times:
            p.update({k: e.get(k) for k in ("t_start_ms", "t_end_ms", "t_peak_ms")})
        if partner:
            p.update({"severity": e["severity"], "severity_note": SEVERITY_NOTE,
                      "confidence": e["confidence"], "peak": e["peak"],
                      "segment_id": e["segment_id"],
                      "matched": e["segment_id"] is not None, "raw_lat": e["lat"],
                      "raw_lon": e["lon"], "length_m": e.get("length_m"),
                      "detail": {"reason": e.get("detail_reason"), "text": e.get("detail_text")},
                      "photo_url": f"/api/photos/{e['photo_path']}" if e.get("photo_path")
                      else None})
        return p

    def geojson(events: list[dict], partner: bool) -> dict:
        feats = []
        for e in events:
            p = event_out(e, partner)
            feats.append({"type": "Feature", "properties": p,
                          "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]}})
        return {"type": "FeatureCollection", "view": "partner" if partner else "public",
                "note": None if partner else PUBLIC_NOTE, **provenance_block(),
                "features": feats}

    def ingest(df: pd.DataFrame, mount: str, device_id: str | None, source: str,
               provenance: str, partner: bool) -> dict:
        bundle = model_or_503()
        try:
            res = process_ride(df, bundle, st.net, mount=mount)
        except ValueError as e:
            raise HTTPException(422, f"could not process the ride: {e}") from None
        with conn() as c:
            rid, events = db.store_ride(c, res, device_id=device_id or db.new_device_id(),
                                        source=source, provenance=provenance, mount=mount)
            stored = {e["id"]: e for e in db.get_ride(c, rid)["events"]}
        out_events = []
        for e in events:
            row = {**stored[e["id"]], "t_peak_ms": e["t_peak_ms"]}
            out_events.append({**event_out(row, partner, with_times=True),
                               "merged_into_existing": e["merged"]})
        return {"ride_id": rid, "provenance": provenance, "summary": res.summary,
                "matched": bool(res.summary.get("gps", {}).get("pct_matched")),
                "events": out_events, **({} if partner else {"note": PUBLIC_NOTE})}

    # ---- endpoints
    @app.get("/api/health")
    def health():
        b = st.model()
        return {"status": "ok", "model_loaded": b is not None,
                "model": b.get("model_name") if b else None,
                "model_provenance": b.get("data_provenance") if b else None,
                "db": s.db_path.name,
                "osm_loaded": st.net is not None,
                "network": None if st.net is None else {
                    "source": st.net.source, "n_segments": len(st.net),
                    "synthetic": st.net.synthetic},
                "hint": None if b else NO_MODEL_HINT}

    @app.get("/api/model")
    def model_info():
        if not s.card_path.exists():
            raise HTTPException(404, NO_MODEL_HINT)
        out = {"model_card": json.loads(s.card_path.read_text())}
        if s.metrics_path.exists():
            out["metrics"] = json.loads(s.metrics_path.read_text())
        if s.notes_path is not None and s.notes_path.exists():
            out["notes"] = json.loads(s.notes_path.read_text())
        out["report_images"] = [f"/reports/{n}" for n in REPORT_IMAGES
                                if s.reports_dir is not None and (s.reports_dir / n).is_file()]
        return out

    @app.get("/api/network")
    def network():
        """The loaded OSM / synthetic network (public): drawn as the base layer of the map."""
        if st.net is None:
            raise HTTPException(404, "no network loaded: run `python -m smartcycle fetch-osm "
                                     "--tracks <dir>` (rides are then shown at raw GPS)")
        if "network_geojson" not in st.extra:
            net = st.net
            feats = [net.segment_geojson(i) for i in range(len(net))]
            for f in feats:
                f["properties"] = {k: f["properties"].get(k) for k in
                                   ("segment_id", "name", "highway", "is_cycleway", "length_m")}
            allc = np.vstack(net.coords)
            st.extra["network_geojson"] = {
                "type": "FeatureCollection", "synthetic": net.synthetic,
                "attribution": None if net.synthetic else
                "Network data © OpenStreetMap contributors (ODbL)",
                "bbox": [float(allc[:, 0].min()), float(allc[:, 1].min()),
                         float(allc[:, 0].max()), float(allc[:, 1].max())],
                "features": feats}
        return st.extra["network_geojson"]

    @app.get("/reports/{name}")
    def report_png(name: str):
        """Read-only report images (confusion matrix, feature importance). PNG names only."""
        d = s.reports_dir
        if d is None or Path(name).name != name or not name.endswith(".png") \
                or name.startswith("."):
            raise HTTPException(404, "report not found")
        p = (d / name).resolve()
        if p.parent != d.resolve() or not p.is_file():
            raise HTTPException(404, "report not found")
        return FileResponse(p, media_type="image/png")

    @app.post("/api/rides/upload-csv")
    async def upload_csv(file: UploadFile = File(...),
                         mount: Literal["handlebar", "pocket"] = Form("handlebar"),
                         device_id: str | None = Form(None, pattern=DEVICE_RE),
                         provenance: Literal["real", "synthetic", "unknown"] = Form("unknown"),
                         x_partner_key: str | None = Header(None)):
        name = Path(file.filename or "").name
        if not name.lower().endswith((".csv", ".txt")):
            raise HTTPException(415, "only .csv ride files are accepted (zip/folder exports: "
                                     "use `python -m smartcycle normalize` first)")
        data = await file.read(MAX_CSV_BYTES + 1)
        if len(data) > MAX_CSV_BYTES:
            raise HTTPException(413, "CSV larger than 20 MB")
        if b"\x00" in data[:4096]:
            raise HTTPException(415, "file is not a text CSV")
        model_or_503()
        try:
            df = canonicalise(read_csv_any(text=decode_bytes(data)), source=name)
        except (ValueError, pd.errors.ParserError) as e:
            raise HTTPException(422, f"could not read the CSV: {e}") from None
        return ingest(df, mount, device_id, f"csv:{name}", provenance,
                      is_partner(x_partner_key))

    @app.post("/api/rides")
    def upload_batch(batch: RideBatch, x_partner_key: str | None = Header(None)):
        model_or_503()
        acc = pd.DataFrame([x.model_dump() for x in batch.samples])
        acc = acc.rename(columns={"t": "timestamp_ms", "ax": "accel_x", "ay": "accel_y",
                                  "az": "accel_z"})
        frames = [acc]
        if batch.gps:
            g = pd.DataFrame([x.model_dump() for x in batch.gps]).rename(columns={
                "t": "timestamp_ms", "lat": "latitude", "lon": "longitude",
                "acc": "gps_accuracy_m", "speed": "speed_mps"})
            frames.append(g)
        raw = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms", kind="stable")
        for c in ("accel_x", "accel_y", "accel_z", "latitude", "longitude", "gps_accuracy_m",
                  "speed_mps"):
            if c not in raw:
                raw[c] = np.nan
        try:
            df = canonicalise(raw.reset_index(drop=True), source="json-batch")
        except ValueError as e:
            raise HTTPException(422, f"could not read the batch: {e}") from None
        return ingest(df, batch.mount, batch.device_id, "json-batch", batch.provenance,
                      is_partner(x_partner_key))

    @app.get("/api/rides")
    def rides():
        with conn() as c:
            return {"rides": db.list_rides(c)}

    @app.get("/api/rides/{ride_id}")
    def ride(ride_id: str, x_partner_key: str | None = Header(None)):
        with conn() as c:
            r = db.get_ride(c, ride_id)
        if r is None:
            raise HTTPException(404, f"ride {ride_id!r} not found")
        partner = is_partner(x_partner_key)
        r["events"] = [event_out(e, partner, with_times=True) for e in r["events"]]
        r.pop("device_id", None)
        return r

    @app.delete("/api/rides/{ride_id}")
    def delete(ride_id: str):
        with conn() as c:
            out = db.delete_ride(c, ride_id, s.photos_dir)
        if out is None:
            raise HTTPException(404, f"ride {ride_id!r} not found")
        return {"deleted": True, **out}

    @app.get("/api/events")
    def events(view: Literal["public", "partner"] = "public",
               bbox: str | None = Query(None, description="minLon,minLat,maxLon,maxLat"),
               type: Literal["Rough", "Pothole", "Irregularity"] | None = None,  # noqa: A002
               x_partner_key: str | None = Header(None)):
        if view == "partner":
            require_partner(x_partner_key)
        bb = None
        if bbox:
            try:
                bb = tuple(float(v) for v in bbox.split(","))
                assert len(bb) == 4 and all(np.isfinite(bb))
            except (ValueError, AssertionError):
                raise HTTPException(400, "bbox must be minLon,minLat,maxLon,maxLat") from None
        with conn() as c:
            evs = db.list_events(c, type, bb)
        return geojson(evs, view == "partner")

    @app.get("/api/partner/segments")
    def segments(x_partner_key: str | None = Header(None)):
        require_partner(x_partner_key)
        with conn() as c:
            rows = db.segment_scores(c, st.net)
        feats = []
        for r in rows:
            if st.net is not None and r["segment_id"] in st.net.index:
                feats.append(st.net.segment_geojson(st.net.index[r["segment_id"]], r))
        return {"type": "FeatureCollection", "score_note": "Score = (sum of point-event "
                "severities + rough metres / 25) per 100 m ridden; uncalibrated heuristic.",
                **provenance_block(), "features": feats}

    @app.get("/api/partner/alerts")
    def partner_alerts(x_partner_key: str | None = Header(None)):
        require_partner(x_partner_key)
        with conn() as c:
            out = db.alerts(c)
        out["alerts"] = [event_out(e, True) for e in out["alerts"]]
        return {**out, **provenance_block()}

    @app.get("/api/partner/priorities")
    def partner_priorities(x_partner_key: str | None = Header(None)):
        require_partner(x_partner_key)
        with conn() as c:
            return {"priorities": db.priorities(c, st.net), **provenance_block()}

    @app.post("/api/events/{event_id}/detail")
    def detail(event_id: str, d: Detail):
        with conn() as c:
            if not db.set_detail(c, event_id, d.reason, d.text):
                raise HTTPException(404, f"event {event_id!r} not found")
            e = db.get_event(c, event_id)
        return {**event_out(e, False), "detail": {"reason": d.reason, "text": d.text}}

    @app.post("/api/events/{event_id}/photo")
    async def photo(event_id: str, file: UploadFile = File(...)):
        with conn() as c:
            if db.get_event(c, event_id) is None:
                raise HTTPException(404, f"event {event_id!r} not found")
        data = await file.read(MAX_PHOTO_BYTES + 1)
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(413, "photo larger than 5 MB")
        try:
            name = save_photo(data, s.photos_dir)
        except PhotoError as e:
            raise HTTPException(415, str(e)) from None
        with conn() as c:
            old = db.set_photo(c, event_id, name)
        if old:
            p = photo_path(s.photos_dir, old)
            if p:
                p.unlink()
        return {"event_id": event_id, "photo_url": f"/api/photos/{name}",
                "metadata_stripped": True, "privacy": PRIVACY_REMINDER}

    @app.get("/api/photos/{name}")
    def get_photo(name: str):
        p = photo_path(s.photos_dir, name)
        if p is None:
            raise HTTPException(404, "photo not found")
        return FileResponse(p, media_type="image/jpeg" if p.suffix == ".jpg" else "image/png")

    if s.web_dir.is_dir():
        app.mount("/", StaticFiles(directory=s.web_dir, html=True), name="web")
    return app


def get_app() -> FastAPI:
    """uvicorn factory: `uvicorn smartcycle.api.app:get_app --factory`."""
    return create_app()
