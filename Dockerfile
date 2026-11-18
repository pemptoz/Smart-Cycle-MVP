# Smart Cycle MVP — READ-ONLY PUBLIC DEMO on the committed REAL data (data/real_public, CC BY 4.0).
# Built from git-tracked files only: the real model is retrained at build time with the exact
# "Reproduce the real results" command (deterministic, seed 0; same metrics as the committed
# reports/), then the 5 real rides are ingested into a fresh SQLite DB. Provenance stays "real".
# SMARTCYCLE_READONLY=1: every write endpoint answers 403. See docs/DEPLOY.md.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SMARTCYCLE_NETWORK=data/osm/cycleways.geojson \
    SMARTCYCLE_MODEL=models/model.joblib \
    SMARTCYCLE_DB=data/smartcycle.db \
    SMARTCYCLE_READONLY=1

COPY pyproject.toml README.md ./
COPY smartcycle ./smartcycle
# same scikit-learn as the committed model card, so the retrained model is the same
RUN pip install --no-cache-dir . "scikit-learn==1.9.1"

COPY web ./web
COPY models ./models
COPY reports ./reports
COPY data/osm ./data/osm
COPY data/real_public ./data/real_public

# real data -> real model (reports to a scratch dir: the committed reports/ stay as in git)
# -> read-only demo DB. The committed model card is replaced by the identical retrained one.
RUN python -m smartcycle train --data-dir data/real_public --mount handlebar --cv blocked \
      --no-speed --final-model rf --reports /tmp/build-reports --out models \
 && rm -rf /tmp/build-reports \
 && python -m smartcycle ingest --dir data/real_public --db data/smartcycle.db

EXPOSE 8000
# PORT is set by most free hosts (Render, Fly, Railway); defaults to 8000 locally.
CMD ["sh", "-c", "python -m smartcycle serve --host 0.0.0.0 --port ${PORT:-8000}"]
