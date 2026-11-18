# Deploy (optional, free tier)

The Docker image is a **synthetic demo**: it contains no real ride, no real database and no real
model (they are not in git and `.dockerignore` excludes them). At build time it generates 6
synthetic rides, trains a synthetic model (`models/synthetic/`) and ingests the rides, so the
dashboard shows the red **SYNTHETIC DEMO DATA** banner. The committed real model card and
metrics are copied but not used by the API (the synthetic model reads `reports/synthetic/`).

## Local
```bash
docker build -t smartcycle-mvp .
docker run --rm -p 8000:8000 smartcycle-mvp      # http://127.0.0.1:8000/
```
Checked on 18 Nov 2026: `/api/health` -> `model_provenance: synthetic`, `/api/events?view=public`
-> synthetic events.

## Free hosting (no account needed locally; hosting itself needs one)
Any free container host that builds a Dockerfile and sets `$PORT` works (e.g. Render "Web
Service" from the Git repo, Fly.io, Railway, Hugging Face Spaces with the Docker SDK). The
container listens on `$PORT` (default 8000) on 0.0.0.0.

## Caveats
- **Ephemeral disk**: free tiers wipe the filesystem on restart / redeploy. Uploaded rides,
  details and photos (SQLite `data/smartcycle.db`, `data/photos/`) are lost; the image rebuilds
  the synthetic demo DB each time. A persistent volume or managed DB would be needed for real use.
- Free instances sleep when idle: the first request can take tens of seconds.
- `PARTNER_KEY` defaults to `demo-partner` and is a demo-only control: set it as a secret env var
  on the host, never commit it. It is not authentication.
- Do not deploy the real DB or real rides publicly (raw GPS = personal data).
