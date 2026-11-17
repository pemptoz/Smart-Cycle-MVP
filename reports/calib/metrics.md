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
| Rule baseline | 0.798 | 0.736 | 0.736 ± 0.067 |
| RandomForest | 0.845 | 0.743 | 0.741 ± 0.027 |
| HistGradientBoosting | 0.855 | 0.645 | 0.655 ± 0.084 |

## Per class (F1: pooled / folds mean ± std)

| Model | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Rule baseline | 0.863 / 0.862 ± 0.089 | 0.672 / 0.711 ± 0.136 | 0.700 / 0.669 ± 0.195 | 0.707 / 0.704 ± 0.095 |
| RandomForest | 0.905 / 0.904 ± 0.065 | 0.835 / 0.865 ± 0.097 | 0.600 / 0.574 ± 0.143 | 0.633 / 0.623 ± 0.051 |
| HistGradientBoosting | 0.930 / 0.929 ± 0.045 | 0.836 / 0.858 ± 0.081 | 0.154 / 0.167 ± 0.236 | 0.659 / 0.667 ± 0.027 |

## Per class precision / recall (pooled)

| Model | Smooth P / R | Rough P / R | Pothole P / R | Irregularity P / R |
|---|---|---|---|---|
| Rule baseline | 0.899 / 0.830 | 0.678 / 0.667 | 0.583 / 0.875 | 0.617 / 0.829 |
| RandomForest | 0.925 / 0.887 | 0.873 / 0.800 | 0.500 / 0.750 | 0.568 / 0.714 |
| HistGradientBoosting | 0.942 / 0.918 | 0.920 / 0.767 | 0.200 / 0.125 | 0.547 / 0.829 |

## Event level (tolerance ±1 s, evaluated distance 0.32 km)

Raw counts only (k detected / n labelled events; false alarms = count): the event numbers are far too small for rates.

> **Pothole has fewer than 10 labelled events (3).**

| Model | Pothole detected | Pothole false alarms | Irregularity detected | Irregularity false alarms |
|---|---|---|---|---|
| Rule baseline | 3/3 | 2 | 13/13 | 4 |
| RandomForest | 3/3 | 2 | 13/13 | 6 |
| HistGradientBoosting | 2/3 | 2 | 13/13 | 9 |

## Per ride (out-of-fold): windows / recall per class, macro-F1

**Rule baseline**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.727 | – | 8 / 0.875 | – | 0.888 |
| irregs | 69 / 0.826 | – | – | 35 / 0.829 | 0.850 |
| rug | – | 60 / 0.667 | – | – | 0.800 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

**RandomForest**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.818 | – | 8 / 0.750 | – | 0.879 |
| irregs | 69 / 0.884 | – | – | 35 / 0.714 | 0.832 |
| rug | – | 60 / 0.800 | – | – | 0.889 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

**HistGradientBoosting**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.857 | – | 8 / 0.125 | – | 0.562 |
| irregs | 69 / 0.928 | – | – | 35 / 0.829 | 0.901 |
| rug | – | 60 / 0.767 | – | – | 0.868 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

## Confusion matrices (pooled out-of-fold, rows = true, row-normalised)

**Rule baseline**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.83 (161) | 0.09 (18) | 0.00 (0) | 0.08 (15) |
| Rough | 0.30 (18) | 0.67 (40) | 0.00 (0) | 0.03 (2) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.88 (7) | 0.12 (1) |
| Irregularity | 0.00 (0) | 0.03 (1) | 0.14 (5) | 0.83 (29) |

**RandomForest**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.89 (172) | 0.03 (6) | 0.00 (0) | 0.08 (16) |
| Rough | 0.18 (11) | 0.80 (48) | 0.00 (0) | 0.02 (1) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.75 (6) | 0.25 (2) |
| Irregularity | 0.09 (3) | 0.03 (1) | 0.17 (6) | 0.71 (25) |

**HistGradientBoosting**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.92 (178) | 0.02 (3) | 0.01 (1) | 0.06 (12) |
| Rough | 0.15 (9) | 0.77 (46) | 0.00 (0) | 0.08 (5) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.12 (1) | 0.88 (7) |
| Irregularity | 0.06 (2) | 0.03 (1) | 0.09 (3) | 0.83 (29) |

## Windows per class

| Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|
| 194 | 60 | 8 | 35 |

_Data provenance: MIXED — REAL data + augmented/synthetic TRAINING windows; test folds real only | 4 rides, 297 windows (Smooth 194, Rough 60, Pothole 8, Irregularity 35)_
