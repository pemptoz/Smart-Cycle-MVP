# Smart Cycle MVP — SYNTHETIC demo image (no real data inside: data/raw, *.db and models/*.joblib
# are not in git and are excluded by .dockerignore). The model is trained on synthetic data at
# build time, so the dashboard shows the SYNTHETIC banner. See docs/DEPLOY.md.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SMARTCYCLE_NETWORK=data/synthetic/routes.geojson \
    SMARTCYCLE_MODEL=models/synthetic/model.joblib \
    SMARTCYCLE_DB=data/smartcycle.db

COPY pyproject.toml README.md ./
COPY smartcycle ./smartcycle
RUN pip install --no-cache-dir .

COPY web ./web
COPY models ./models
COPY reports ./reports
COPY data/osm ./data/osm

# synthetic data -> synthetic model -> demo DB (all inside the image, regenerated on each build)
RUN python -m smartcycle synth --out data/synthetic --rides 6 --seed 0 --plot '' \
 && python -m smartcycle train --data-dir data/synthetic \
 && python -m smartcycle ingest --dir data/synthetic --db data/smartcycle.db

EXPOSE 8000
# PORT is set by most free hosts (Render, Fly, Railway); defaults to 8000 locally.
CMD ["sh", "-c", "python -m smartcycle serve --host 0.0.0.0 --port ${PORT:-8000}"]
