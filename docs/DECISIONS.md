# Decisions log

Small decisions are taken by Claude Code and logged here; big architecture changes are agreed with the user first.
P00 plan validated on Sun 15 Nov 2026 (with amendments A1–A8, included below).

## Schedule
- Sun 15 Nov (evening): P01 foundation, P02 ML pipeline, P02b labelling tool — synthetic data only.
- Mon 16 Nov: real data collection + P03 (geo, SQLite, FastAPI).
- Tue 17 Nov: P05 (real data: ingest, label, retrain, diagnose) + P04 dashboard.
- Wed 18 Nov: P06 polish (+ Android PWA recorder if time).

## Environment
- Python 3.12, no `uv` on the machine -> `python3 -m venv .venv && pip install -e .[dev]`.

## Data loading (P01)
- Alias table on normalised column names (lowercase, units and punctuation stripped). Missing columns -> clear error listing columns found and expected.
- CSV separator and decimal comma are sniffed. Phyphox `Time (s)` (relative) converted to ms.
- CLI `normalize` also accepts a folder or .zip with separate Phyphox sensor files (e.g. `Accelerometer.csv` + `Location.csv`), merged on time with `merge_asof`. GPS columns stay empty between fixes in canonical files (never interpolated there).
- **(A7)** The upload API accepts **canonical-or-aliased CSV only** (<= 20 MB); zip/folder support exists only in the CLI `normalize`.
- GPS-only rows (fix on its own row, accel empty) are attached to the nearest accel sample (merge_asof, tolerance max(100 ms, 2×median dt)); the GPS-only row is dropped, values never interpolated. Duplicate timestamps: keep first.
- Time detection: |median| > 1e11 -> epoch ms, 1e9–1e11 -> epoch s, otherwise relative with unit from the header (`(s)`, `_ms`), else median step < 1 -> seconds.
- Labels: case-insensitive, anything outside the 4 classes raises. If a file has a `label` column AND `labels.csv` has intervals for that ride, `labels.csv` wins (warning logged). Overlapping intervals: Pothole > Irregularity > Rough > Smooth; uncovered samples = NA; `t_end_s` exclusive.
- Phyphox "Linear Acceleration" (no gravity) is deliberately NOT aliased; the error message says to export acceleration with g. Phyphox `Velocity (m/s)` maps to `speed_mps`.
- Pandas 3.x installed: `label` uses the nullable `string` dtype.
- `inspect` reports columns, recognised aliases, duration, estimated sampling rate (median dt), jitter, GPS fix ratio, |a| range, and **(A2)** the count of stationary windows.

## Synthetic data (P01)
- Rides of 3–8 min, random sampling rate per ride **50–100 Hz** (P01 brief; supersedes the earlier 40–200 Hz idea), timestamp jitter (~5 % dt) and 0–2 gaps of 0.3–1 s, ~1 Hz GPS (5 % fixes dropped) with AR(1)-correlated 3–6 m noise along hand-made polylines in the Dublin bbox, accuracy 3–10 m, speed 3–7 m/s (clipped 2.5–8), pedalling 1–1.5 Hz with coasting phases, rides start with 3–6 s at rest + 1–2 stops (for A2 stationary windows).
- Mount: handlebar = gravity on +Z with a fixed tilt < 10°; pocket = random fixed rotation + slow drift (2–6°), road components low-passed (12 Hz) and damped ×0.8, larger pedalling (leg) component.
- Overlap: unlabelled bumps and small swerves inside Smooth/Rough, weak Rough patches, mild potholes; `--hard` doubles distractor rate, weakens events, makes Smooth noisier.
- Layouts (to exercise the loader): rides `i % 3 == 2` are plan-style (`Timestamp (ms)` epoch-ms, `Accel_X (m/s²)`, GPS on the nearest accel row, no accuracy/speed columns); the others are Phyphox-like (`Time (s)` relative, `Acceleration with g x (m/s^2)`, GPS fixes on their own rows); rides 1 and 7 use `;` + decimal comma.
- Ride files carry **no label column**; labels live only in `labels.csv` (Smooth/Rough segment intervals + event intervals ±0.4 s, overlapping; resolved by priority Pothole > Irregularity > Rough > Smooth). This exercises the merge path used for real data.
- `routes.geojson` is flagged `"synthetic": true` (invented geometry, NOT OSM). `MANIFEST.json` = {provenance: synthetic, seed, version, ...}. Fixed epoch base (16 Nov 2026 08:00 UTC) -> byte-identical output for a given seed.
- `data/synthetic/` is gitignored (regenerable); `reports/synth_examples.png` is committed.

## Preprocessing and windows (P02)
- Sampling rate estimated from median dt, resampled (linear) to `target_hz` = 50; gaps > 0.5 s split the signal (no interpolation across gaps).
- **(A4)** Gravity = zero-phase low-pass ~0.5 Hz (Butterworth + `filtfilt`), fallback 3 s rolling median. Not a 1 s moving average.
- Orientation-robust channels: vertical dynamic (projection of a − g on ĝ), horizontal dynamic (norm of the residual), |a| − 9.81. A "fixed-axes" feature set (x, y, z dynamic) is kept as an ablation.
- **(A3)** Default `window_s = 1.0`, `step_s = 0.5`; `window_s = 2.0` only as an ablation.
- Window label: if a Pothole/Irregularity interval overlaps the window by >= 0.2 s -> that label (priority Pothole > Irregularity); otherwise majority Rough/Smooth; windows without label coverage are dropped.
- **(A2)** Stationary windows (mean GPS speed < 1 m/s) are **excluded from training, evaluation and reported metrics**; at inference they are predicted `Smooth`. They are counted in `inspect` and `validate-labels`.
- Features per channel: std, RMS, peak-to-peak, max |.|, kurtosis, skew, crest factor, max jerk, FFT band energies (0.5–3, 3–10, 10–25 Hz), plus mean GPS speed (NaN -> training-fold median).

## Models and evaluation (P02)
- Rule baseline (thresholds on vertical peak, horizontal peak, vertical RMS; thresholds learned on training folds only), RandomForest, HistGradientBoosting; `class_weight="balanced"`; fixed hyperparameters, **no tuning**.
- CV grouped by ride: `StratifiedGroupKFold` (k = min(5, n_rides)), `LeaveOneGroupOut` if fewer than 5 rides. Runtime assertion + test that train/test ride ids never intersect. Augmentation (random rotation, noise) only inside training folds.
- **(A6)** Macro-F1 and the confusion matrix are computed on **pooled out-of-fold predictions**, plus per-ride tables; `zero_division=0` for folds with missing classes. Event-level metrics: event recall for Pothole and Irregularity, false alarms per km. The report states that picking the best of 3 models on CV is slightly optimistic.
- Final model = best CV macro-F1, refit on all data -> `models/model.joblib` + `models/model_card.json` (`data_provenance`, counts, features, CV metrics, limitations).

## Labelling tool (P02b)
- `web/label.html`, standalone (works from `file://`), **(A7)** reads only the canonical CSV (no alias logic in JS).
- Signal plot with zoom/pan, mini-map, keys 1–4 for the 4 labels, import/export `labels.csv` (`ride_id,t_start_s,t_end_s,label`).
- 3-tap sync detection kept as an **optional helper**.

## Geo (P03)
- **(A1)** `fetch-osm` (Overpass API, bbox of tracks + 300 m) builds two layers in one GeoJSON with an `is_cycleway` property:
  - cycleways (`is_cycleway=true`): `highway=cycleway`, `cycleway[:*]=lane|track`, `bicycle=designated`;
  - ordinary roads/paths (`is_cycleway=false`): residential, tertiary, secondary, unclassified, service, path, footway, track — where `bicycle != no`.
  Ways are split into ~50 m sections (id `way:index`). A small committed fallback extract is kept.
- Local equirectangular projection (no pyproj).
- Map matching: KD-tree candidates + exact point-to-segment projection; cost = distance + heading penalty + continuity bonus + **(A1)** small preference for cycleways. **(A5)** The heading penalty is **undirected** (angle difference modulo 180°). Max radius = max(20 m, 2 × GPS accuracy), otherwise unmatched. Reported: median/p90 snap distance, % matched.
- Events: consecutive non-Smooth windows of the same label merged; Pothole/Irregularity located at the peak time; Rough as an interval over sections. Multi-ride aggregation: same label, same section, < 10 m -> same anomaly.
- Severity 1–5 = **uncalibrated heuristic** from peak amplitude bins; stated in docstrings, API (`severity_note`) and UI. Section score = sum of severities.

## Storage and API (P03)
- SQLite, 4 tables: `rides`, `detections`, `anomalies` (recomputed on ingest/delete), `photos`. Network sections read from the GeoJSON, not stored in the DB.
- Endpoints: `GET /api/health`, `GET /api/model`, `POST /api/rides`, `GET /api/rides`, `GET /api/rides/{id}`, `DELETE /api/rides/{id}`, `GET /api/anomalies?view=public|partner`, `GET /api/partner/summary`, `GET /api/network`, `POST /api/anomalies/{id}/photos`, `GET /api/photos/{id}`, static `/` -> `web/`.
- **(Q4 / A8)** Partner access: `X-Partner-Key` header checked against env `PARTNER_KEY` (default `demo-partner`). Responses without a valid key never contain severity or scores. Documented as a demo-grade control, not real auth.
- Security/privacy: CSV upload <= 20 MB, photos <= 5 MB opened with Pillow and **re-encoded to JPEG without EXIF**, UUID file names, parameterised SQL, anonymous random `device_id`, delete-ride endpoint, warning not to photograph people/plates.

## Frontend (P04)
- Dashboard reuses the website tokens/UX (Outfit, light/dark), Leaflet 1.9.4 vendored + OSM tiles with attribution, public vs partner view, alerts, priority repairs, attach photo, provenance badge, "Replay a ride" (upload CSV -> real pipeline -> animation).
- Android PWA recorder: `DeviceMotionEvent.accelerationIncludingGravity` + `watchPosition` + Wake Lock, exports canonical CSV and uploads it. Secure context via Chrome desktop USB port forwarding to `http://localhost:8000`.

## Answers to the P00 questions
- Q1 (Phyphox experiment + sample lines): **open** — not yet provided; default = support single CSV and folder/zip.
- Q2 (ground-truth method): **open** — default = 3-tap sync + controlled passes over pre-scouted, photographed defects + notes, labelling afterwards in `label.html`.
- Q3 (deliverable format + time limit): **open** — default = local live demo + CSV replay fallback; free deployment only if time remains.
- Q4: use `X-Partner-Key` (see above).
- Q5: no external datasets.

## Cut list
Cut from the top first if we fall behind.
1. Dockerfile / free deployment
2. 1D-CNN
3. PWA service worker / installability, then the whole PWA (Phyphox + CSV upload is enough)
4. HMM map matching (stay on nearest + undirected heading + continuity)
5. **(A8)** Free-text detail/reason on an anomaly ("user can detail afterwards")
6. Fine multi-ride aggregation (fallback: 1 detection = 1 point)
7. Ablations: robust vs fixed-axes, 1 s vs 2 s windows, handlebar vs pocket (keep robust, 1 s)
8. Mini-map in `label.html` (keep the signal plot)
9. Replay animation (keep static display of results)
10. Photo gallery (keep upload + EXIF stripping)

**Never cut:** loaders + tests, grouped CV + model card + provenance, stationary-window exclusion, basic map matching, API ingest/anomalies/delete with partner-key filtering, Leaflet map public/partner, CSV upload path.
