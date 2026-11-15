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

## Preprocessing and windows (P02) — as implemented
- **Sampling rate**: steps > 5× the median are treated as gaps and excluded, then fs = 1 / MEAN step. The median alone is biased when timestamps are whole ms (80 Hz → 12/13 ms steps → median says 83 Hz). Resampled to `fs` = 50 Hz (configurable): zero-phase Butterworth anti-alias (0.4·fs, order 4) when the native rate is higher, then linear interpolation. Grid samples inside gaps (> 0.5 s) are flagged invalid; no window may contain one.
- **(A4)** Gravity = zero-phase 0.5 Hz Butterworth (`sosfiltfilt`); rolling-median (3 s) fallback for very short signals. Channels: `a_vert = acc·ĝ − mean|g|`, `a_hmag = |acc − (acc·ĝ)ĝ|`, `a_dev = |acc| − mean|acc|` (= `invariant` set), `ax, ay, az = acc − g` (= `fixed` set).
- **GPS**: real fixes only; fixes with accuracy > 30 m are dropped (counted in `prep.info`). Speed comes from `speed_mps` when present, otherwise from a centred haversine difference over ±2 fixes followed by a 5-fix rolling median. Known limit: on synthetic plan-style rides (no speed column, 3–6 m correlated GPS noise) this speed is too noisy to detect short stops (0–4 stationary windows instead of 15–50). Real Phyphox exports have a speed column.
- **(A3)** `window_s = 1.0`, `step_s = 0.5` by default.
- **Window label** (the brief supersedes the 0.2 s of P00): a Pothole/Irregularity interval overlapping by **>= 0.3 s** wins (Pothole > Irregularity); otherwise majority Rough/Smooth by duration. A window with labelled coverage < 50 % is dropped, and so is a window where the only coverage is a short event overlap. Intervals are rebuilt as runs from per-sample labels (after `merge_labels` priority), so a `label` column and `labels.csv` go through the same path.
- **(A2)** Stationary windows (mean GPS speed < 1 m/s) are dropped from training and evaluation. At `predict` they are output with `pred = "stationary"` and never form events (equivalent to "not reported" on the map). Without any GPS speed nothing is dropped and a warning is logged.
- **Features** (the brief's list supersedes the P00 list): per channel mean, std, rms, min, max, ptp, skew, kurtosis, mean |jerk|, zero-crossing rate, band energies 0–3/3–8/8–15/15–25 Hz (clipped to Nyquist), dominant frequency. Extra features: speed (NaN → ride median + `speed_missing` indicator, then a training-fold median imputer inside the model pipeline), rms(a_vert)/max(speed, 1), peak ratio a_hmag/|a_vert|, relative peak time. `invariant` gives 50 features and `invariant+fixed` gives 98.

## Models and evaluation (P02) — as implemented
- Models share one interface: an sklearn `Pipeline(SimpleImputer(median), clf)`.
  - `RuleBaseline`: Pothole if peak |a_vert| > T1, elif peak a_hmag > T2 → Irregularity, elif rms a_vert > T3 → Rough, else Smooth. Candidate thresholds are 100 quantiles (0.30–0.995) per feature plus "disabled", tried by **coordinate ascent** (3 rounds) to maximise macro-F1 on the **training fold only**. A full 12³ grid was tried first: it was too coarse and missed clean separations in a unit test.
  - RF as specified (`n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0`).
  - HGB (`max_iter=200, lr=0.1, early_stopping=False, random_state=0`) with `compute_sample_weight("balanced")` passed at fit time.
  - No tuning anywhere. No augmentation (so nothing to leak).
- **CV**: `StratifiedGroupKFold(k=min(5, n_rides), shuffle, seed 0)` grouped by ride. This deviates from plain GroupKFold; it is still ride-grouped and gives better rare-class balance per fold. A runtime assertion runs on every fold. With < 3 rides: blocked time-series CV (3 contiguous blocks per ride, train windows within 2·window_s of the test block removed) and a LOUD stderr warning.
- **Metrics**: pooled out-of-fold accuracy / macro-F1 / per-class P-R-F1 / confusion matrix (counts + row-normalised), plus fold mean ± std. In per-fold tables a class absent from a test fold is NaN and skipped, not counted as 0. Event level: a labelled Pothole/Irregularity is recalled if a predicted window of that class overlaps it ±1 s. A false alarm is a merged predicted event with no labelled event of the same class within ±1 s. Distance = Σ window speed × step_s over evaluated windows.
- **Ablations** (`--ablate`, RF only, to keep runtime around 2 min): invariant vs invariant+fixed, 1 s vs 2 s windows, with vs without speed features, and macro-F1 per mount on the baseline out-of-fold predictions.
- **Final model**: RandomForest by default (`--final-model rf|hgb|rule|best`), refit on all moving labelled windows. `models/model.joblib` (gitignored) bundles model + config + feature columns + classes + provenance. `models/model_card.json` is committed.
- **Events** (`smartcycle/geo/events.py`, reused in P03): windows of the same class whose gap is ≤ 1 s are merged. Pothole/Irregularity become point events at the window with the highest peak magnitude |(a_vert, a_hmag)|; Rough becomes a stretch.
- **Synthetic difficulty**: RF macro-F1 = 0.883 (< 0.98), so the generator was NOT made harder.

## Labelling tool (P02b) — as implemented
- `web/label.html`, single file, no dependency, works from `file://`; **(A7)** reads only the canonical CSV (no alias logic in JS; a non-canonical file gets an error listing its columns and the `normalize` command).
- Time base = seconds from the first sample after sort + de-duplication of timestamps (keep first), same as the loader's `t_s`.
- Pure logic lives in `<script id="label-core">` (no DOM); `tests/js/label_core.test.mjs` extracts that block from the HTML and runs it in `node:vm`, so the tool stays one file and the logic is unit-tested.
- Segments are kept **non-overlapping**: a new label cuts the overlapping parts of older ones (split into left/right remainders); touching same-label segments are merged; slivers < 0.01 s dropped. So an event inside a Smooth stretch splits it: exported files never rely on the loader's overlap priority.
- Event = cursor ± 0.4 s clamped to the ride. `E` = Pothole, `Shift+E` = Irregularity (buttons too).
- Candidates: peaks of |accel_z − 1 s moving mean| ("vertical", meaningful in handlebar mode) and of hypot(x, y deviations) ("lateral"), above an adjustable threshold (default 3 m/s²), greedy non-max suppression at 1.5 s. Candidate times are a navigation aid only, never labels.
- Time offset (s): applied **only on export** (t + offset, clamped to [0, duration], empty rows dropped), shown as a badge and dashed ghost outlines on the plot. It replaces the planned 3-tap sync detector (not implemented: cut for time; the offset field is enough to align a video clock).
- Export times have 3 decimals. Merge: load an existing labels.csv, rows of the current ride_id are replaced, others kept, sorted by ride_id then t_start.
- Autosave in `localStorage` (key `smartcycle-label:<ride_id>:<n_samples>`), wrapped in try/catch; restore is offered on load. `beforeunload` warns while changes are not exported.
- Colours: website tokens; Smooth `--lo`, Rough `--mid`, Pothole `--hi`, Irregularity purple `#7b4fc9` / `#b49cf0` (dark), since the site has no 4th status colour.
- Test hook: `label.html?src=<url>&selftest=1` (over http) loads a CSV, runs a scripted session through the real keyboard handlers and writes the export into `<pre id="selftest">` (headless Chrome check).

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
