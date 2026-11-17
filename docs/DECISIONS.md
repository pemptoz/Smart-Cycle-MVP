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
- **(A7)** The upload API accepts **canonical-or-aliased CSV only** (<= 20 MB); zip/folder support exists only in the CLI `normalize`. P03 adds the phone JSON batch (`POST /api/rides`) as requested by the P03 brief.
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

## Geo (P03) — as implemented
The P03 brief supersedes the P00 notes for this section (Viterbi instead of greedy matching,
brief's OSM selectors incl. primary roads, ids `seg_<wayid>_<k>`, r = max(15 m, 2 x accuracy)).
- **OSM** (`smartcycle/geo/osm.py`, `fetch-osm`): bbox of all fixes of the ride CSVs under
  `--tracks` (file or dir) + 300 m; one Overpass query (urllib, User-Agent, timeout 90 s, 2 retries
  across overpass-api.de / kumi mirror). `is_cycleway=true`: highway=cycleway; path/footway/track/
  pedestrian with bicycle designated|yes; any `cycleway[:left|right|both]` not in
  {no, none, **separate**} (separate = lane mapped as its own way, which we fetch anyway).
  `is_cycleway=false`: residential, tertiary, secondary, primary, unclassified, service.
  Ways are split at junction nodes (needed so connectivity = shared endpoints), then into
  n = ceil(L/100) equal pieces -> max 100 m, 50–100 m for long ways. In dense city centre most
  pieces are shorter (Dublin test extract: median 21 m, p95 81 m) because of junctions; accepted.
  Cached: no download if the file exists unless `--force`. Overpass failure -> exit 3 and a message;
  the pipeline then runs with `matched=false`.
- `data/osm/cycleways.geojson` committed: small real extract (666 segments, 224 kB) around
  3 points near Trinity College (proof of the Overpass path). Refetch with `--force` on real
  tracks in P05.
- **Network loader** (`geo/network.py`): segmented files are used as is; plain LineStrings (the
  SYNTHETIC `routes.geojson`) are segmented on the fly as `seg_<route_id>_<k>`. Local
  equirectangular projection around the network centroid. Connectivity = endpoints equal at 1e-7°.
- **Matching** (`geo/matching.py`, numpy only, constants in `MATCH_CFG`): candidates within
  r = max(15 m, 2 x accuracy), top 8; none -> nearest within 40 m; none -> unmatched (breaks the
  chain). Emission 0.5 (d/σ)², σ = max(8 m, accuracy), −0.3 for cycleways. Transition 0 same,
  0.5 connected, 3 other, +10 if the snapped jump > 2·max(speed·dt, raw jump) + 30 m. Missing
  accuracy -> 8 m. Brute-force point-to-edge projection in chunks (fast enough at city scale).
  Test: noisy synthetic tracks recover >= 90 % of the true segments (measured 96.8 %).
- **Events placement**: lat/lon at the peak window (interpolated GPS), snapped onto the segment of
  the nearest-in-time matched fix (<= 5 s), else `snap_point`. Events without any GPS are dropped
  and counted (`n_events_without_gps`).
- **Severity** (`geo/severity.py`, `SEVERITY_BINS`, UNCALIBRATED HEURISTIC, stated in docstring,
  API `severity_note` and to be shown in the UI): Pothole on peak |a_vert| [<6, 6–9, 9–13, 13–18,
  >=18]; Irregularity on peak a_hmag [<3, 3–5, 5–8, 8–12, >=12]; Rough on mean window rms(a_vert)
  [<0.5, 0.5–0.8, 0.8–1.2, 1.2–2, >=2] m/s² (from the SYNTHETIC Rough distribution, 10/50/90 % =
  0.41/0.67/1.15; revisit on real data). The three magnitudes are read from existing window
  features (`a_vert__max/min`, `a_hmag__max`, `a_vert__rms`) in `predict_ride`; training unchanged.
  `confidence` (mean model probability) is stored separately. `peak` = the magnitude used.
- **Coverage / score**: distance between consecutive matched fixes (gap <= 5 s) attributed to the
  first fix's segment; `rough_m` = the part inside Rough events. Segment score =
  (Σ point-event severities + rough_m / 25) / max(covered_m / 100, 1); bands >= 6 hi "Repair soon",
  >= 3 mid "Needs attention", else lo "Good" (website thresholds). Heuristic.

## Storage and API (P03) — as implemented
Supersedes the P00 table/endpoint list (brief's names used).
- SQLite tables: `rides`, `events` (`seq` autoincrement -> id `A-0001`, never reused; extra columns
  `length_m` for Rough, `observed_at` = ride start + t_peak when timestamps are epoch, else NULL and
  `created_at` is used as the date), **`event_reports(event_id, ride_id, t_*_ms, severity)`** (extra
  table: which rides back an event — needed for dedup counts, per-ride replay times and GDPR
  deletion), `coverage(ride_id, segment_id, covered_m, rough_m)`, `ride_track(ride_id, geojson)`
  (matched track <= 1 pt/s, <= 3000 pts, with `t_ms` and `matched`). A JSON copy of the ride
  summary is kept in `rides.summary`.
- Dedup: Pothole/Irregularity only, same type + same (non-null) segment + <= 12 m between snapped
  points -> `reports` = number of distinct rides, severity = max, peak = max. Rough stretches are
  never merged. Unmatched events are never merged.
- Delete ride: events backed only by this ride are deleted with their photo files; merged events
  lose the report (owner ride reassigned, severity = max of the remaining reports).
- Provenance: rides carry `real|synthetic|unknown`. API uploads default to **unknown** (the server
  cannot know; the PWA will send `real`), `ingest` auto-detects `MANIFEST.json` (synthetic) — single
  files outside such a dir are `unknown`, `--dir` without manifest is `real` (same rule as
  `load_dataset`). Every events/partner response has `model_provenance` + `data_provenance`
  (none|real|synthetic|unknown|mixed).
- Alerts: severity >= 4 within 10 days of "now", top 4 by severity then date. "Now" = wall clock,
  unless no event falls in [now − 10 d, now] (old or future-dated demo data): then the latest
  event date, flagged `reference: "latest_event"` with a note.
- Partner key: `X-Partner-Key` vs env `PARTNER_KEY` (default `demo-partner`), constant-time
  compare; 401 otherwise. Ride detail and upload responses include severity only with the key.
- `GET /api/rides` never returns `device_id` (would let anyone link rides of one phone).
- Model loaded lazily (a model trained after the server started is picked up); no model -> 503 with
  the `train` hint. Errors are JSON `{"error": ...}` (validation: `{"error", "details"}`).
- Photos: magic bytes + Pillow verify, JPEG/PNG only, <= 5 MB, decompression-bomb cap 40 MP,
  EXIF orientation applied then the image is rebuilt from raw pixels (drops EXIF/GPS/ICC/text),
  max 1280 px, random 32-hex name; served by `GET /api/photos/{name}` (strict pattern), URL only in
  the partner view.
- `web/index.html` is a placeholder until P04 (links to /docs, /api/health, label tool).
- Demo rides: `synth --out data/demo_rides --rides 3 --minutes 2 --seed 7` (seed ≠ training seed),
  gitignored; real demo rides decided in P05/P06.

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
4. ~~HMM map matching~~ (implemented in P03: Viterbi HMM-lite)
5. ~~**(A8)** Free-text detail/reason on an anomaly~~ (implemented in P03: `POST /api/events/{id}/detail`)
6. Fine multi-ride aggregation (fallback: 1 detection = 1 point)
7. Ablations: robust vs fixed-axes, 1 s vs 2 s windows, handlebar vs pocket (keep robust, 1 s)
8. Mini-map in `label.html` (keep the signal plot)
9. Replay animation (keep static display of results)
10. Photo gallery (keep upload + EXIF stripping)

**Never cut:** loaders + tests, grouped CV + model card + provenance, stationary-window exclusion, basic map matching, API ingest/anomalies/delete with partner-key filtering, Leaflet map public/partner, CSV upload path.

## Real data (P05) — facts and class mapping
Decided with the user BEFORE looking at the signals (17 Nov).
- 5 Phyphox recordings (iPhone, 100 Hz accel, GPS ~1 Hz, `Velocity (m/s)` speed column), each
  ONE continuous recording, no gaps: `bump` 58 s, `irregs` 62 s, `pocket` 74 s, `rug` 38 s,
  `smooth` 37 s (~4.5 min in total). No second collection will happen.
- Mapping: `bump` -> **Pothole** (proxy: vertical shocks from speed bumps / regular bumps; there
  are NO real potholes in the data), `irregs` -> **Irregularity** (crossing irregular objects /
  obstacles; NOT swerve or braking), `rug` -> **Rough** (gravel), `smooth` -> **Smooth** (central
  clean part only), `pocket` (phone in a coat pocket, mixed events) -> **orientation test only**,
  never in the main training/evaluation. Mount: handlebar for all except `pocket`.
- The class name stays `Pothole` in code and data; UI text says "Pothole / bump".
- **Limitations stated everywhere results are shown**: real potholes and real swerve/braking
  events are absent; each recording is ~one class, slow (mean moving speed 1.7–2.3 m/s) and at a
  different speed -> class/speed confound risk; one session, one phone, one rider.
- Area: university campus in south Dublin (lat ~53.306–53.309, lon ~-6.226 to -6.218): expect
  paths / footways / service roads, not only cycleways.
- `data/raw/rides.csv` lists the rides with `file = canonical/<id>.csv` (normalised copies).
  `data/raw/` is gitignored (raw GPS traces are not committed).

## GPS policy (P05 A1)
- Fixes with accuracy > 30 m are dropped (`--max-gps-accuracy`, <= 0 keeps all), counted per ride
  and printed by `validate-labels` (real data: irregs 3, smooth 2, others 0).
- Speed = Phyphox column (all 5 rides). Stationary windows (< 1 m/s) are removed from training,
  evaluation and reported metrics; at inference they are output as **`Smooth`** with
  `stationary=True` and NaN confidence and never form events (user's rule; supersedes the earlier
  `pred = "stationary"` pseudo-label — map-wise identical: no event).
- Map-matching radius r = max(**20 m**, 2 x accuracy) (was 15 m). A 1031 m-accuracy fix is dropped
  by the policy; if kept, its emission is near-flat (sigma = accuracy) and its neighbours stay on
  the true segment (tested).

## Evaluation protocol for the real data (P05 A2)
- Ride-grouped CV is INVALID here: each recording is ~one class, so a test ride holds a class
  never seen in training. `--cv grouped` and `--cv logo` now **refuse** (clear error naming the
  classes and rides) when any class appears in < 2 rides (grouped keeps its < 3-rides fallback
  for synthetic tests only).
- `--cv blocked` (`--block-s 15`, `--purge-s 2`): block = ride + floor(t_start / block_s) (time
  from the first sample, same grid for every window length); StratifiedGroupKFold over blocks with
  k = min(5, n_blocks, min over classes of the number of blocks holding that class), refused if
  k < 2. Split seeds 0..49 are tried in order and the first one giving every class in every test
  fold is used — the choice uses labels only, never scores, so it is not tuning. Otherwise a LOUD
  warning lists the missing classes per fold. Purge: any training window within `purge_s` of a
  test window of the same ride is removed (asserted on every fold).
- Header printed on stderr, stdout and at the top of `metrics.md`: "blocked CV on a tiny
  single-session dataset: optimistic, no ride-level generalisation claim". Protocol (k, seed,
  blocks per class, per-fold train/purged/test counts and classes) goes into `metrics.json`.
- Metrics stay pooled out-of-fold (+ fold mean ± std); new per-ride table (windows and recall per
  class, macro-F1) in `metrics.md`, `per_ride` in `metrics.json`.
- `--mount handlebar|pocket|all` selects the rides used for CV/training. Main real run:
  `--mount handlebar` (pocket is an orientation test only).

## Experiments (P05 A3, `evaluate --ablate`, RandomForest, same CV protocol as the main run)
- (i) speed: all speed features (`speed`, `speed_missing`, `rms_vert_per_speed`) vs none, plus a
  **speed-only diagnostic** (`speed`, `speed_missing` only; `rms_vert_per_speed` contains
  acceleration so it is excluded there). Per-class speed table (n, mean, median, std).
  CONFOUNDING WARNING when speed-only macro-F1 > 0.40 (chance ~0.25) or the class mean speeds
  differ by more than x1.20.
- (ii) `invariant` vs `invariant+fixed`; (iii) 1 s vs 2 s windows (re-windowed; block grid is
  time-based so the same blocks are used).
- (iv) orientation test: RF trained on ALL handlebar windows of the dataset (+ augmentation /
  synthetic if enabled, train side only), tested on the labelled moving windows of the pocket
  ride(s); metrics only over the classes labelled in pocket (listed in the output), for
  `invariant` and `invariant+fixed`. Skipped with a note when pocket has no labels.
- `evaluate` uses the FULL dataset for (iv) even with `--mount handlebar`.

## Model card and provenance (P05 A4)
- Card fields added: `git_dirty` (`git status --porcelain` non-empty; a stderr WARNING is printed
  when training on a dirty tree), `cv_protocol` (CVConfig + split details: k, seed, blocks, folds),
  `augmentation` (null or the augmentation parameters), `n_augmented_windows`,
  `n_synthetic_windows`, `mount_filter`.
- `data_provenance` = `real` | `mixed` | `synthetic`. Real data + augmentation and/or synthetic
  training windows = **mixed** (user's rule: even augmentation-only is mixed). Reports and the card
  say "REAL data + augmented/synthetic TRAINING windows; test folds real only"; synthetic keeps
  the "NOT REAL RESULTS" warning. `metrics.json` also has `test_data_provenance`.
- API: `/api/health`, `/api/model` and every events/partner response pass `mixed` through
  unchanged (tested). The dashboard is still the P04 placeholder: the provenance badge does not
  exist yet; P04 must render real / mixed / synthetic distinctly.

## Augmentation (P05 A5, `smartcycle/features/augment.py`, `--augment`)
- Applied inside each CV fold to the TRAINING windows only (seed = fold index), and to all windows
  for the final model; test windows are never augmented and OOF predictions exist only for the
  original real windows.
- Allowed samples per ride = union of the samples of the fold's training windows (after purge),
  minus test-window samples (belt and braces). Every source range of every augmented window is
  inside that set (tested on every fold of a blocked CV).
- Refactor for this: `decompose` keeps the gravity estimate (`gx, gy, gz`), channels are computed
  by `channels_from(dyn, g, mean|g|, mean|acc|)`, features by `features_from_arrays`; windows carry
  `i0` (first grid sample). An identity transform reproduces the real window's features exactly
  (tested).
- Per training window, 1 transformed copy: time-warp f ~ U(0.9, 1.1) (span of W·f samples centred
  on the window, only if fully allowed, resampled to W; speed feature x f), rotation of dyn AND g
  about the window's mean gravity by U(0, 2pi) (invariant channels unchanged — tested — fixed axes
  rotated), amplitude x U(0.85, 1.15), Gaussian noise 0.05 x per-axis std.
- Event paste: 2 x (number of training Pothole/Irregularity windows) new windows; a training
  Pothole/Irregularity interval (>= 0.3 s, cropped to 0.8 W around its peak if longer), mean-removed,
  Hann-tapered, scaled U(0.85, 1.15), added at a random position inside a training Smooth/Rough
  window; label = event class, speed = background's.
- Provenance of anything trained with augmentation on real data = **mixed** (card `augmentation`
  = the parameters, `n_augmented_windows`).

## Calibrated synthetic data (P05 A6, `smartcycle/synth/calibrate.py`)
- Calibration from real labelled windows: per-axis Welch spectrum and std of dyn in the gravity
  frame (vertical / lateral / longitudinal) for Smooth and Rough; per labelled Pothole /
  Irregularity interval: peak |a_vert|, peak a_hmag, duration, dominant frequency; gravity vectors
  and speeds per class.
- Generation: shaped noise with the class spectrum (events on the Smooth spectrum); Pothole =
  damped vertical oscillation with real amplitude / frequency / duration (x U(0.85, 1.15));
  Irregularity = real-like obstacle response (vertical + 0.5 x horizontal) or, for 50 %, a
  lateral swerve / longitudinal braking pulse (0.3–0.8 s) labelled **Irregularity** (plan
  definition; such events are ABSENT from the real data — stated in the MANIFEST).
- **No leakage**: inside CV (`--extra-train calibrated`) the calibration is redone per fold from
  the fold's TRAINING windows only (tested: changing test windows does not change the calibration)
  and windows are generated in memory, as many per class as the fold has training windows. Test
  folds stay real-only. For the final model the calibration uses all real labelled windows.
- `synth --calibrate-from data/raw/canonical --labels data/raw/labels.csv` (the parent's
  `rides.csv` is used) writes `data/synthetic_calibrated/` (gitignored): canonical rides WITHOUT
  GPS, labels.csv, rides.csv, MANIFEST `provenance: synthetic-calibrated` (read as synthetic).
  These files are for inspection; training uses the in-memory path.
- Time-boxed: a plausible model fitted on ~4 min of data, not validated against held-out rides.

## Part B reporting rules (P05, user rules given with the label counts, fixed BEFORE results)
- Labels (17 Nov): Pothole = 3 events in `bump` + 1 in `pocket`; Irregularity = 13 events, all in
  `irregs`; Rough = `rug` only; Smooth from `bump`, `irregs`, `smooth`.
- Event-level results are raw counts only (`k/n` detected, false alarms as a count + evaluated km),
  never rates; reports state that Pothole has fewer than 10 events. Card `cv_metrics` stores
  `events_detected` ("k/n") and `false_alarms` counts instead of recall rates.
- DIAGNOSTIC 3-class experiment (Smooth / Rough / Impact = Pothole + Irregularity), same folds as
  the 4-class run (splits computed on the 4-class labels). It shows what the data can support; it
  does NOT replace the 4-class model or the 4 product classes.
- Pocket test: per-class window counts and event counts only for the classes labelled in pocket
  (Pothole only); no overall score.
- **Speed features**: the main run and the final model use `--no-speed`; the `(i) WITH speed
  features` row is shown next to it. Speed is put back only if adding it raises the RF pooled
  macro-F1 by more than **0.05** (threshold fixed before looking at the results).
- Clean tree: the user's deletion of `data/label_test/syn_00.csv` (a P02b label-tool fixture, not
  used by any test) is committed; `web/labels*.csv` (an empty stray export) is gitignored.
