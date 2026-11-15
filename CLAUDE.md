# Smart Cycle — MVP for the "AI for Engineering" course

Smartphone accelerometer + GPS -> ML classifier of road quality on cycleways -> map-matched anomalies -> dashboard (public view / partner-city view). Deadline **Wed 18 Nov 2026**. Scope is small on purpose: **it must work end-to-end; it does not need to be perfect.**

Full context: `docs/context/CONTEXT.md` (read it at the start of every session). Progress memory: `docs/PROGRESS.md` (read first, update last). Original docs in `docs/context/` are READ-ONLY.

## Language
- Talk to the user in **French**. Everything in the repo (code, comments, UI text, docs, commit messages) in **English**.

## Priorities (cut from the bottom if time runs out)
P1 ML pipeline + API + map-matching  >  P2 dashboard with real map  >  P3 Android PWA recorder (always keep the "upload / replay a CSV ride" path working)  >  optional: Dockerfile / free deployment, 1D-CNN.

## Stack (all free / open source)
Python 3.11+, numpy, pandas, scipy, scikit-learn, joblib, matplotlib, FastAPI + uvicorn, python-multipart, Pillow, pytest, httpx. SQLite (stdlib). Frontend: vanilla HTML/CSS/JS, no build step, Leaflet (vendored in `web/vendor/`) + OpenStreetMap tiles (keep attribution). Package name `smartcycle`, installed with `pip install -e .` (use `uv` if available, otherwise venv). No paid APIs, no accounts.

## Layout
```
smartcycle/            # python package
  io/                  # loaders (canonical, plan-style, Phyphox-style), column aliases
  synth/               # synthetic ride generator
  features/            # preprocessing, windows, features
  models/              # rule baseline, RF, HGB, evaluation, model card
  geo/                 # OSM fetch, map matching, events grouping, severity
  api/                 # FastAPI app, DB, photo handling
  cli.py               # `python -m smartcycle <command>`
web/                   # index.html (dashboard), record.html (PWA), label.html (labelling tool), site/ (original marketing page), assets/, vendor/
data/{raw,synthetic,osm,demo_rides,photos}/   # raw = REAL data only (gitignored except small demo rides)
models/  reports/  docs/  tests/
```

## CLI contract (use exactly these names)
`python -m smartcycle` + : `synth` · `inspect <csv>` · `normalize <raw.csv> --out <csv>` · `validate-labels --data-dir DIR` · `train --data-dir DIR` · `evaluate --data-dir DIR` · `predict <csv>` · `fetch-osm --tracks DIR` · `ingest <csv>` · `serve` · `report`

## Data contract
- **Canonical ride CSV**: `timestamp_ms, accel_x, accel_y, accel_z, latitude, longitude, gps_accuracy_m, speed_mps, label` — accel in m/s² INCLUDING gravity; `timestamp_ms` may be epoch-ms or ride-relative ms; lat/lon/accuracy/speed are empty between GPS fixes; `label` optional.
- Loader must also accept: plan-style names (`Timestamp (ms), Accel_X, Accel_Y, Accel_Z, Latitude, Longitude, Ground_Truth_Label`) and Phyphox-style names (e.g. `Time (s)`, `Acceleration x (m/s^2)`, `Latitude (°)`, `Longitude (°)`, `Speed (m/s)`, `Horizontal Accuracy (m)`) through an alias table, and fail with a clear message listing the columns it found when something is missing. The real export format is NOT known yet: never assume.
- **Labels**: only `Smooth`, `Rough`, `Pothole`, `Irregularity`. Either a `label` column or `labels.csv` with `ride_id,t_start_s,t_end_s,label` (seconds from the first sample of the ride).
- **Ride metadata** `data/raw/rides.csv`: `ride_id,file,mount(handlebar|pocket),device,notes`. Real rides live in `data/raw/rides/*.csv`, labels in `data/raw/labels.csv`, normalised copies in `data/raw/canonical/`. Synthetic data lives in `data/synthetic/` with a `MANIFEST.json`. OSM network: `data/osm/cycleways.geojson` (override with env `SMARTCYCLE_NETWORK`).
- Sampling rate is UNKNOWN: never hardcode it. Estimate from median dt, then resample to a configurable target (default 50 Hz).

## Non-negotiable rules
1. Only the 4 plan labels. Do not invent Crack/Faulting/Debris classes.
2. **Never present synthetic results as real.** Every model has `models/model_card.json` with `data_provenance` = `real` | `synthetic` | `mixed`; API and UI must show it; reports must state it.
3. **No leakage:** cross-validation is grouped by ride. Never split windows of the same ride across train/test. Never tune on the test fold. Augmentation only on training folds.
4. **Never say "done / works / passes" without running the command and showing the output.** If something cannot be run, say so.
5. Do not touch or "fix" tests to make them pass unless the test itself is wrong — explain why.
6. Privacy: anonymous random device ids, no accounts, strip EXIF from photos, provide a delete-ride endpoint, warn users not to photograph people/plates.
7. Severity (1–5) is an uncalibrated heuristic; label it so in code docs and UI.
8. Keep code small, readable, typed where cheap, with tests for core logic (loaders, resampling, windowing/labelling, no-leakage, map matching, API happy path).
9. Security basics on the API: validate upload type and size, no path traversal, parameterised SQL, no secrets in git.
10. Decide small things yourself and log them in `docs/DECISIONS.md`; ask the user before big architecture changes.

## Workflow
- One phase per session. Start each session: read this file, `docs/context/CONTEXT.md`, `docs/PROGRESS.md`, `git log --oneline -10`.
- Plan first (plan mode), then implement; small commits per logical step; end each phase with: tests green, a short demo command, `docs/PROGRESS.md` updated, commit.
- Use independent sub-agents for review only (see prompts P90), not for writing core code.
- Do not start the next phase unprompted.
