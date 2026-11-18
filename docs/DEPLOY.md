# Deploy (optional, free tier)

The Docker image is a **read-only public demo on the committed real data**. It is built from
git-tracked files only:

1. `data/real_public/` (5 real rides, labels, start times; CC BY 4.0, published with the
   recorder's written consent, no device metadata) and `data/osm/cycleways.geojson` (ODbL).
2. At build time the real model is **retrained** with the "Reproduce the real results" command
   (`train --data-dir data/real_public --mount handlebar --cv blocked --no-speed --final-model rf`,
   seed 0, scikit-learn pinned to the version of the committed model card), because
   `models/model.joblib` is not in git. The retrained card replaces the committed one inside the
   image only; the committed `reports/` are served unchanged (same metrics, see README).
3. The 5 rides are ingested into a fresh SQLite DB (`data/smartcycle.db`), with recording times
   from `rides.csv` (`started_at_utc`).
4. `SMARTCYCLE_READONLY=1`: every POST / DELETE (`/api/rides/upload-csv`, `/api/rides`,
   `/api/events/{id}/detail`, `/api/events/{id}/photo`, `DELETE /api/rides/{id}`) answers
   **403**; GET endpoints are unchanged. The dashboard hides upload, detail and photo controls
   and shows the banner "Read-only public demo". Model and data provenance stay **real**.

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
4. **Environment** -> add `PARTNER_KEY` = a value of your choice (the partner view key; never
   commit it). `SMARTCYCLE_READONLY=1` is already set in the image; do not override it.
5. **Health check path**: `/api/health`. Create the service; the first build takes a few
   minutes (pip install + training on ~4.5 min of rides).
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
  it as a secret env var on the host.
- Only `data/real_public/` is published. Never deploy `data/raw/` (raw Phyphox exports with
  device metadata) or a local DB that contains uploads.
- Writable deployment (uploads, photos): unset `SMARTCYCLE_READONLY` only with a persistent
  volume, a real authentication layer and a privacy review; out of scope for this MVP.
