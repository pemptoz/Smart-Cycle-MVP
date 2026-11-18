# Smart Cycle MVP — READ-ONLY PUBLIC DEMO on the committed REAL data (data/real_public, CC BY 4.0).
# Built from git-tracked files only: the committed real model (models/model.joblib, scikit-learn
# pinned in pyproject.toml) is served as is, and the 5 real rides are ingested into a fresh
# SQLite DB. Provenance stays "real". The model can be rebuilt with the "Reproduce the real
# results" command of the README (same metrics). SMARTCYCLE_READONLY=1: every write endpoint
# answers 403, so the published demo partner key (shown as a hint) only unlocks reading.
# See docs/DEPLOY.md.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SMARTCYCLE_NETWORK=data/osm/cycleways.geojson \
    SMARTCYCLE_MODEL=models/model.joblib \
    SMARTCYCLE_DB=data/smartcycle.db \
    SMARTCYCLE_READONLY=1

COPY pyproject.toml README.md ./
COPY smartcycle ./smartcycle
# scikit-learn is pinned in pyproject.toml to the version that saved models/model.joblib
RUN pip install --no-cache-dir .

COPY web ./web
COPY models ./models
COPY reports ./reports
COPY data/osm ./data/osm
COPY data/real_public ./data/real_public

# committed real model -> read-only demo DB (fails the build if the model cannot be loaded)
RUN python -m smartcycle ingest --dir data/real_public --db data/smartcycle.db

EXPOSE 8000
# PORT is set by most free hosts (Render, Fly, Railway); defaults to 8000 locally.
CMD ["sh", "-c", "python -m smartcycle serve --host 0.0.0.0 --port ${PORT:-8000}"]
