# PROGRESS (Claude Code: update this at the end of EVERY session)

Deadline: Wed 18 Nov 2026. Data collection: Mon 16 Nov.

## Status
| Phase | Status | Notes |
|---|---|---|
| P00 plan | done | Plan validated with amendments A1–A8, see docs/DECISIONS.md |
| P01 foundation + synthetic data | done | Loaders (3 layouts + Phyphox folder/zip), inspect, normalize, synth generator, 34 tests green |
| P02 ML pipeline | done | Preprocess/windows/features, rule+RF+HGB, ride-grouped CV, events, reports, model card; 60 tests green |
| P02b labelling tool | done | `web/label.html` (single file, offline), node tests 13/13, pytest 62 green, headless Chrome e2e + `validate-labels` OK |
| P03 geo + backend | done | OSM fetch, Viterbi matching, events/severity/coverage, SQLite, FastAPI (public/partner), photos, ingest/serve; 82 tests green |
| P04 dashboard + PWA | todo | |
| P05 real data | todo | |
| P06 polish + report assets | todo | |
| P90 independent review | todo | |

## Decisions log (also see docs/DECISIONS.md)
- 15 Nov: P00 plan validated. Schedule: Sun P01+P02+P02b (synthetic) · Mon data collection + P03 · Tue P05 + P04 · Wed P06 (+ PWA).
- Key amendments: two OSM layers (cycleways + ordinary roads, cycleway preference); stationary windows (<1 m/s) excluded from train/eval/metrics; 1 s windows / 0.5 s step; 0.5 Hz zero-phase gravity filter; undirected heading penalty; pooled out-of-fold metrics + event-level metrics; API accepts CSV only; X-Partner-Key for partner view.

## Known issues / next steps
- Open questions: Q1 Phyphox experiment + sample lines, Q2 ground-truth method, Q3 deliverable format + time limit (defaults in docs/DECISIONS.md).
- P01 (15 Nov): package skeleton, `smartcycle.io` (alias table, CSV sniffing, time-unit detection, GPS-row merge, labels merge, `load_dataset` + provenance), `inspect`, `normalize`, `synth` + `reports/synth_examples.png`. Other CLI commands print "not implemented yet, phase PXX".
- Known limits: synthetic signals are a plausible model, not validated against real rides; real Phyphox export format still unknown (Q1) -> run `inspect` on the first real file Monday and extend the alias table if needed.
- `data/synthetic/` is gitignored: regenerate with the synth command below (deterministic per seed).
- P02 (15 Nov): `smartcycle/features/` (preprocess, windows, features, build), `smartcycle/models/` (rule, zoo, evaluation, report, card, predict, labels_check, train), `smartcycle/geo/events.py` (window → events, reused in P03). CLI: `validate-labels`, `train`, `evaluate [--ablate]`, `predict`.
- P02 results on **SYNTHETIC** data only (12 rides, 7901 moving windows, ride-grouped 5-fold CV, pooled out-of-fold): rule baseline macro-F1 0.690 · RF 0.883 (±0.016 across folds) · HGB 0.881. RF event recall: Pothole 0.99, Irregularity 0.97, false alarms ≈ 0.7 / 0.9 per km. Ablations: +fixed axes 0.885, no speed 0.882, 2 s windows 0.900; handlebar 0.906 vs pocket 0.858. These numbers say nothing about real roads.
- Known limits: haversine speed (when there is no speed column) misses short stops on noisy GPS. `train --ablate` takes ~2 min, `train` ~1.5 min (permutation importance). `predict` on a training ride is an in-sample demo, not an evaluation.
- P02b (15 Nov): `web/label.html` labelling tool — canonical CSV in, `labels_<ride_id>.csv` / merged `labels.csv` out. Plot (accel_z + toggles x/y/|a|/speed), minimap, GPS mini-plot, drag-select + 1–4, events E/⇧E (±0.4 s), fill unlabelled, candidates N/P, segment table, undo/redo, autosave, time offset on export. Logic unit-tested in node (`tests/js/label_core.test.mjs`); export fixture checked by `tests/test_label_tool_export.py` (`load_labels`, `interval_issues`, `merge_labels`). Headless Chrome self-test on normalised `syn_01` -> export passes `validate-labels`.
- Known limits: file input / drag-drop / clipboard only checked by hand (headless check uses `?src=` + scripted keys); 3-tap sync detector not implemented (offset field instead); default candidate threshold (3 m/s²) gives ~1 candidate / 6 s on synthetic rides — tune it on real data.
- P03 (15 Nov): `geo/network.py` (segments, connectivity), `geo/osm.py` + `fetch-osm` (Overpass, cached; real Dublin extract 666 segments committed), `geo/matching.py` (Viterbi HMM-lite, 96.8 % segments recovered on noisy synthetic tracks), `geo/severity.py` (bins, score, bands — uncalibrated heuristic), `pipeline.py` (one ride -> events/coverage/track), `api/db.py` (SQLite, dedup, GDPR delete, alerts/priorities), `api/app.py` + `api/photos.py` (all endpoints of the brief + `GET /api/rides`, `GET /api/photos/{name}`), CLI `ingest`, `serve`. Architecture: `docs/ARCHITECTURE.md`.
- P03 acceptance run on SYNTHETIC data (synthetic network, model trained on synthetic): 3 demo rides (seed 7) ingested via CLI, CSV upload and JSON batch; 100 % fixes matched, snap median 2–3.6 m; all endpoints curled. `fetch-osm` against the real Overpass API worked (666 segments).
- Known limits: OSM segments are short in the city centre (split at every junction, median 21 m); severity bins and the Rough rms bins come from synthetic data; alerts use wall clock (future-dated synthetic rides only appear once the reference switches to "latest event"); `web/index.html` is a placeholder until P04; the partner key is demo-only.
- Next: Monday — record rides, `normalize` each raw file into `data/raw/canonical/`, label with `web/label.html`, merge into `data/raw/labels.csv`, run `inspect` + `validate-labels`. Then P03 (waiting for the user's go).

## Last commands that worked (copy-paste)
```
python3 -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
pytest -q
python -m smartcycle synth --out data/synthetic --rides 12 --seed 0      # [--hard] [--minutes 2]
python -m smartcycle inspect data/synthetic/rides/syn_00.csv
python -m smartcycle normalize data/synthetic/rides/syn_01.csv --out /tmp/syn_01_canonical.csv
python -m smartcycle inspect /tmp/syn_01_canonical.csv --labels data/synthetic/labels.csv --ride-id syn_01
python -m smartcycle validate-labels --data-dir data/synthetic
python -m smartcycle train --data-dir data/synthetic [--ablate] [--final-model rf|hgb|rule|best]   # reports/ + models/
python -m smartcycle evaluate --data-dir data/synthetic --ablate
python -m smartcycle predict data/synthetic/rides/syn_03.csv [--out preds.csv] [--all]
node --test tests/js/label_core.test.mjs                                   # label tool logic
xdg-open web/label.html                                                   # labelling tool (file://), load a `normalize` output
python -m smartcycle normalize data/raw/rides/X.csv --out data/raw/canonical/X.csv   # before labelling
python -m smartcycle fetch-osm --tracks data/raw/canonical --buffer-m 300 [--force]  # real OSM network
python -m smartcycle synth --out data/demo_rides --rides 3 --minutes 2 --seed 7 --plot ''
SMARTCYCLE_NETWORK=data/synthetic/routes.geojson python -m smartcycle ingest --dir data/demo_rides   # [--db X.db]
SMARTCYCLE_NETWORK=data/synthetic/routes.geojson python -m smartcycle serve --port 8000   # http://127.0.0.1:8000/docs
curl -F file=@data/demo_rides/rides/syn_01.csv -F provenance=synthetic localhost:8000/api/rides/upload-csv
curl -H 'X-Partner-Key: demo-partner' 'localhost:8000/api/events?view=partner'
```
