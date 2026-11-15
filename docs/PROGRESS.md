# PROGRESS (Claude Code: update this at the end of EVERY session)

Deadline: Wed 18 Nov 2026. Data collection: Mon 16 Nov.

## Status
| Phase | Status | Notes |
|---|---|---|
| P00 plan | done | Plan validated with amendments A1–A8, see docs/DECISIONS.md |
| P01 foundation + synthetic data | todo | |
| P02 ML pipeline | todo | |
| P02b labelling tool | todo | |
| P03 geo + backend | todo | |
| P04 dashboard + PWA | todo | |
| P05 real data | todo | |
| P06 polish + report assets | todo | |
| P90 independent review | todo | |

## Decisions log (also see docs/DECISIONS.md)
- 15 Nov: P00 plan validated. Schedule: Sun P01+P02+P02b (synthetic) · Mon data collection + P03 · Tue P05 + P04 · Wed P06 (+ PWA).
- Key amendments: two OSM layers (cycleways + ordinary roads, cycleway preference); stationary windows (<1 m/s) excluded from train/eval/metrics; 1 s windows / 0.5 s step; 0.5 Hz zero-phase gravity filter; undirected heading penalty; pooled out-of-fold metrics + event-level metrics; API accepts CSV only; X-Partner-Key for partner view.

## Known issues / next steps
- Open questions: Q1 Phyphox experiment + sample lines, Q2 ground-truth method, Q3 deliverable format + time limit (defaults in docs/DECISIONS.md).
- Next: P01 (waiting for the user's go).

## Last commands that worked (copy-paste)
