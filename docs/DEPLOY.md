# Deploy (optional, free tier)

The Docker image is a **read-only public demo on the committed real data**. It is built from
git-tracked files only:

1. `data/real_public/` (5 real rides, labels, start times; CC BY 4.0, published with the
   recorder's written consent, no device metadata) and `data/osm/cycleways.geojson` (ODbL).
2. The committed real model `models/model.joblib` (scikit-learn pinned in `pyproject.toml` to
   the version recorded in the model card) is served as is, with the committed card and
   `reports/`. It is not retrained at build time; to rebuild it, run the README "Reproduce the
   real results" command (same metrics). If it could not be loaded, the build fails at the
   ingest step (and a running API reports `loaded: false` instead of crashing).
3. The 5 rides are ingested into a fresh SQLite DB (`data/smartcycle.db`), with recording times
   from `rides.csv` (`started_at_utc`).
4. `SMARTCYCLE_READONLY=1`: every POST / DELETE (`/api/rides/upload-csv`, `/api/rides`,
   `/api/events/{id}/detail`, `/api/events/{id}/photo`, `DELETE /api/rides/{id}`) answers
   **403**, whatever the delete token or partner key; GET endpoints are unchanged. The dashboard
   hides upload, detail and photo controls and shows the banner "Read-only public demo". Model
   and data provenance stay **real**.
5. Partner key: with the default `demo-partner` (no `PARTNER_KEY` set), the read-only demo
   shows the hint "demo partner key: demo-partner" in the partner prompt (it only unlocks
   reading). A writable server refuses to start with the default key unless
   `SMARTCYCLE_ALLOW_DEMO_KEY=1`.

## Local
```bash
docker build -t smartcycle-public .
docker run --rm -p 8000:8000 smartcycle-public      # http://127.0.0.1:8000/
curl -s localhost:8000/api/health                    # model_provenance real, readonly true
curl -s -o /dev/null -w '%{http_code}\n' -X DELETE localhost:8000/api/rides/x   # 403
```

## Free hosting: Render (exact steps)
Any free container host that builds a Dockerfile and sets `$PORT` works (Render, Fly.io,
Railway, Hugging Face Spaces with the Docker SDK); the container listens on `0.0.0.0:$PORT`
(default 8000). On Render:

1. Push the repository to GitHub (public or private).
2. <https://dashboard.render.com> -> sign in (free account) -> **New** -> **Web Service** ->
   connect the GitHub repository, branch `main`.
3. Render detects the `Dockerfile`: language **Docker**, root directory empty, instance type
   **Free**. No build or start command (the Dockerfile `CMD` is used; Render sets `PORT`).
4. **Environment**: `SMARTCYCLE_READONLY=1` is already set in the image (keep it; you may add
   it explicitly in Render's Environment tab). Optional: `PARTNER_KEY` = a secret of your
   choice (never commit it); without it the demo key `demo-partner` is used and shown as a
   hint, which is fine for a read-only demo. Never set `SMARTCYCLE_ALLOW_DEMO_KEY` on a host.
5. **Health check path**: `/api/health`. Create the service; the first build takes a few
   minutes (pip install + ingest of the 5 rides).
6. Check the public URL `https://<service>.onrender.com/`: read-only banner, real model badge,
   13 events on the UCD campus; `/api/health` says `"readonly": true`.

Every push to `main` redeploys (auto-deploy can be turned off in the service settings).

## QR code for the final URL (offline)
```bash
pip install -e '.[qr]'                                   # optional dependency: segno
python scripts/make_qr.py https://<service>.onrender.com/ --out docs/qr.png   # or .svg
```
No web service is called; the QR code encodes only the URL. Test it with a phone camera before
printing it on a slide or poster.

## Caveats
- **Ephemeral disk**: free tiers wipe the filesystem on restart / redeploy. This does not matter
  in read-only mode (nothing is written); the DB is rebuilt from `data/real_public/` at each
  build.
- Free instances sleep when idle: the first request can take tens of seconds (Render free: about
  a minute after 15 min idle).
- `PARTNER_KEY` defaults to `demo-partner` and is a demo-only control, not authentication: set
  it as a secret env var on the host (required for any writable deployment).
- Only `data/real_public/` is published. Never deploy `data/raw/` (raw Phyphox exports with
  device metadata) or a local DB that contains uploads.
- Writable deployment (uploads, photos): unset `SMARTCYCLE_READONLY` only with a persistent
  volume, a real authentication layer and a privacy review; out of scope for this MVP.
