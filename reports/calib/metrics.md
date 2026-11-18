# Smart Cycle — classifier evaluation

> **blocked CV on a tiny single-session dataset: optimistic, no ride-level generalisation claim.**

**Data provenance: MIXED — REAL data + augmented/synthetic TRAINING windows; test folds real only | 4 rides, 297 windows (Smooth 194, Rough 60, Pothole 8, Irregularity 35)**

- Blocked CV: blocks of 15 s inside each ride (14 blocks; per class {'Irregularity': 4, 'Pothole': 3, 'Rough': 3, 'Smooth': 11}), StratifiedGroupKFold over blocks, k = 3, split seed 0 (first seed giving every class in every test fold), training windows within 2 s of a test window removed (purge).
  - fold 0: train 146 (purged 39), test 112, classes ['Irregularity', 'Pothole', 'Rough', 'Smooth']
  - fold 1: train 183 (purged 20), test 94, classes ['Irregularity', 'Pothole', 'Rough', 'Smooth']
  - fold 2: train 186 (purged 20), test 91, classes ['Irregularity', 'Pothole', 'Rough', 'Smooth']
- TRAINING folds extended with: augmentation=False, extra_train=calibrated (extra windows per fold [146, 183, 186]); test folds contain real windows only.
- CV: blocked, 3 folds
- Windows: 1.0 s, step 0.5 s, resampled to 50 Hz; features: `invariant`, speed features: False
- Stationary windows (< 1.0 m/s) dropped: 1
- Pooled = metrics on all out-of-fold predictions; mean ± std = across folds (classes absent from a test fold are skipped for that fold).
- Picking the best of 3 models on the same CV is slightly optimistic.

## Overall

| Model | Accuracy (pooled) | Macro-F1 (pooled) | Macro-F1 (folds) |
|---|---|---|---|
| Rule baseline | 0.801 | 0.756 | 0.761 ± 0.121 |
| RandomForest | 0.825 | 0.711 | 0.716 ± 0.050 |
| HistGradientBoosting | 0.835 | 0.623 | 0.626 ± 0.081 |

## Per class (F1: pooled / folds mean ± std)

| Model | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Rule baseline | 0.866 / 0.864 ± 0.087 | 0.667 / 0.698 ± 0.121 | 0.778 / 0.752 ± 0.256 | 0.713 / 0.731 ± 0.158 |
| RandomForest | 0.903 / 0.903 ± 0.071 | 0.781 / 0.815 ± 0.168 | 0.571 / 0.546 ± 0.110 | 0.588 / 0.600 ± 0.030 |
| HistGradientBoosting | 0.915 / 0.916 ± 0.052 | 0.788 / 0.825 ± 0.200 | 0.143 / 0.133 ± 0.189 | 0.644 / 0.628 ± 0.073 |

## Per class precision / recall (pooled)

| Model | Smooth P / R | Rough P / R | Pothole P / R | Irregularity P / R |
|---|---|---|---|---|
| Rule baseline | 0.904 / 0.830 | 0.684 / 0.650 | 0.700 / 0.875 | 0.596 / 0.886 |
| RandomForest | 0.915 / 0.892 | 0.911 / 0.683 | 0.462 / 0.750 | 0.500 / 0.714 |
| HistGradientBoosting | 0.913 / 0.918 | 0.932 / 0.683 | 0.167 / 0.125 | 0.538 / 0.800 |

## Event level (tolerance ±1 s, evaluated distance 0.32 km)

Raw counts only (k detected / n labelled events; n predicted event units; false alarms = unmatched units): the event numbers are far too small for rates. Matching: one-to-one: a labelled event is matched by at most one predicted event unit of the same class whose peak lies within the tolerance of the labelled interval; merged predicted events are cut into units of at most 2 s, each unmatched unit is a false alarm.

> **Pothole has fewer than 10 labelled events (3).**

| Model | Pothole detected | Pothole predicted | Pothole false alarms | Irregularity detected | Irregularity predicted | Irregularity false alarms |
|---|---|---|---|---|---|---|
| Rule baseline | 3/3 | 4 | 1 | 11/13 | 18 | 7 |
| RandomForest | 3/3 | 5 | 2 | 11/13 | 20 | 9 |
| HistGradientBoosting | 2/3 | 5 | 3 | 13/13 | 23 | 10 |
| _reference: whole-ride Pothole_ | 3/3 | 75 | 72 | 0/13 | 0 | 0 |
| _reference: whole-ride Irregularity_ | 0/3 | 0 | 0 | 13/13 | 75 | 62 |
| _reference: all Smooth_ | 0/3 | 0 | 0 | 0/13 | 0 | 0 |

Reference rows: degenerate predictors (every evaluated window predicted as that class) scored with the same metric, to calibrate expectations.

## Per ride (out-of-fold): windows / recall per class, macro-F1

**Rule baseline**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.727 | – | 8 / 0.875 | – | 0.888 |
| irregs | 69 / 0.826 | – | – | 35 / 0.886 | 0.866 |
| rug | – | 60 / 0.650 | – | – | 0.788 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

**RandomForest**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.831 | – | 8 / 0.750 | – | 0.882 |
| irregs | 69 / 0.884 | – | – | 35 / 0.714 | 0.835 |
| rug | – | 60 / 0.683 | – | – | 0.812 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

**HistGradientBoosting**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.870 | – | 8 / 0.125 | – | 0.565 |
| irregs | 69 / 0.913 | – | – | 35 / 0.800 | 0.872 |
| rug | – | 60 / 0.683 | – | – | 0.812 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

## Confusion matrices (pooled out-of-fold, rows = true, row-normalised)

**Rule baseline**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.83 (161) | 0.09 (17) | 0.00 (0) | 0.08 (16) |
| Rough | 0.28 (17) | 0.65 (39) | 0.00 (0) | 0.07 (4) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.88 (7) | 0.12 (1) |
| Irregularity | 0.00 (0) | 0.03 (1) | 0.09 (3) | 0.89 (31) |

**RandomForest**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.89 (173) | 0.02 (3) | 0.00 (0) | 0.09 (18) |
| Rough | 0.23 (14) | 0.68 (41) | 0.00 (0) | 0.08 (5) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.75 (6) | 0.25 (2) |
| Irregularity | 0.06 (2) | 0.03 (1) | 0.20 (7) | 0.71 (25) |

**HistGradientBoosting**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.92 (178) | 0.01 (2) | 0.01 (1) | 0.07 (13) |
| Rough | 0.23 (14) | 0.68 (41) | 0.00 (0) | 0.08 (5) |
| Pothole | 0.00 (0) | 0.12 (1) | 0.12 (1) | 0.75 (6) |
| Irregularity | 0.09 (3) | 0.00 (0) | 0.11 (4) | 0.80 (28) |

## Windows per class

| Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|
| 194 | 60 | 8 | 35 |

_Data provenance: MIXED — REAL data + augmented/synthetic TRAINING windows; test folds real only | 4 rides, 297 windows (Smooth 194, Rough 60, Pothole 8, Irregularity 35)_
