# Experiments log (REAL data, P05 Part B, 17 Nov 2026)

All runs: `data/raw`, `--mount handlebar` (bump, irregs, rug, smooth), **blocked CV on a tiny
single-session dataset: optimistic, no ride-level generalisation claim** (15 s blocks, purge 2 s,
k = 3 — limited by Pothole and Rough having only 3 blocks each; split seed 0 has every class in
every test fold), 1 s windows / 0.5 s step, `invariant` features, fixed hyperparameters, no tuning.
Test folds contain REAL windows only. 297 moving labelled windows: Smooth 194, Rough 60,
Pothole 8, Irregularity 35. Events: **Pothole 3 (fewer than 10)**, Irregularity 13 (one ride).
Evaluated distance 0.318 km. Event results are counts (k/n), never rates.

## E1 — Main result (real only, no speed features) — `reports/`
Why: the reference result; speed excluded by the rule fixed before the run (DECISIONS).
`python -m smartcycle evaluate --data-dir data/raw --mount handlebar --cv blocked --no-speed --ablate`

| Model | Accuracy | Macro-F1 pooled | Macro-F1 folds | Pothole events | Pothole FA | Irreg. events | Irreg. FA |
|---|---|---|---|---|---|---|---|
| Rule baseline | 0.801 | 0.739 | 0.739 ± 0.067 | 3/3 | 2 | 13/13 | 4 |
| RandomForest | 0.879 | 0.778 | 0.790 ± 0.063 | 3/3 | 1 | 13/13 | 4 |
| HistGradientBoosting | 0.832 | 0.607 | 0.606 ± 0.034 | 2/3 | 7 | 13/13 | 6 |

RF pooled F1: Smooth 0.925, Rough 0.852, Pothole 0.588 (8 windows), Irregularity 0.747.
RF confusion (counts, rows = true): Smooth 179/5/1/9 · Rough 11/49/0/0 · Pothole 0/0/5/3 ·
Irregularity 3/1/3/28. Main confusion: Pothole <-> Irregularity (both are vertical shocks of
similar magnitude here) and Rough -> Smooth.

## E2 — Experiments (RF, same protocol)
| Experiment | Macro-F1 pooled | Pothole ev. | Irreg. ev. | Note |
|---|---|---|---|---|
| baseline (no speed) | 0.778 | 3/3 | 13/13 | |
| (i) WITH speed features | 0.787 | 3/3 | 13/13 | +0.009 < 0.05 -> speed stays OUT |
| (i) speed ONLY | 0.342 | 0/3 | 13/13 | confound diagnostic |
| (ii) invariant+fixed | 0.786 | 3/3 | 13/13 | +0.008, not worth orientation dependence |
| (iii) 2 s windows | 0.788 | 3/3 | 13/13 | +0.010, Rough F1 drops 0.852 -> 0.784 |
| DIAGNOSTIC 3 classes | 0.885 | Impact 16/16 (1 FA) | | Smooth 0.930, Rough 0.857, Impact 0.867 |

Speed per class (m/s, mean): Smooth 2.198, Rough 1.673, Pothole 2.680, Irregularity 2.494 —
x1.60 between classes -> CONFOUNDING WARNING printed. Speed alone reaches only 0.342 (chance
~0.25), and removing speed costs 0.009: the model does not rely on speed much, but the class /
speed / recording confound cannot be ruled out with this data.

(iv) Orientation test (train handlebar -> pocket): **only Pothole is labelled in pocket (1 event,
2 windows); only that class is meaningful and no overall score is reported.** Pothole windows
0/2 (both predicted Irregularity), events 0/1, for both `invariant` and `invariant+fixed`.

## E3 — Secondary: real + augmentation (`reports/aug/`, provenance MIXED)
`... --no-speed --augment --reports reports/aug` — extra train windows per fold 186/239/242.
RF macro-F1 0.755 (folds 0.752 ± 0.058), Pothole 2/3 (2 FA), Irregularity 13/13 (4 FA). Rough
recall improves (56/60), Pothole/Irregularity do not. Rule 0.709, HGB 0.695.

## E4 — Secondary: real + calibrated synthetic (`reports/calib/`, provenance MIXED)
`... --no-speed --extra-train calibrated --reports reports/calib` — calibration redone per fold on
training windows; extra windows per fold 146/183/186. RF macro-F1 0.743 (0.741 ± 0.027), Pothole
3/3 (2 FA), Irregularity 13/13 (6 FA). Rule 0.736, HGB 0.645.

## Diagnosis (in the requested order)
1. Label / signal alignment: OK. Every labelled event contains a peak |a_vert| of 8–30 m/s²
   (ride p95 3.0–4.3) and the ±1 s peak sits inside the interval (two Irregularity intervals,
   29.2–30.0 s and 33.7–34.5 s, have their strongest ±1 s peak just outside / in a neighbour).
2. Class counts per block: Pothole 8 windows in 3 blocks, Rough 60 windows in 3 blocks -> k = 3.
3. Speed confound: see E2 (warning, small effect).
4. Window size: 2 s helps Pothole slightly, hurts Rough; kept 1 s (no tuning on test folds).
5. Orientation: pocket Pothole missed (0/1), predicted Irregularity.
6. Overfitting gap: RF in-sample macro-F1 0.961 vs blocked-CV 0.778 (in-sample value now stored in `reports/insample.json`, computed from the saved model by `report --data-dir data/raw`; see `docs/RESULTS.md`).

## Decision (B4)
Final model = **RandomForest, real data only, no speed features, invariant, 1 s windows** (best
pooled CV on real blocks: 0.778; augmentation 0.755 and calibrated synthetic 0.743 did not help)
-> provenance **real**. HGB is clearly worse on 8 Pothole windows. The 3-class diagnostic says
the data supports "impact vs no impact" much better than Pothole vs Irregularity.
