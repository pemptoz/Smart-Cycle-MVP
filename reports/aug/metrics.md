# Smart Cycle — classifier evaluation

> **blocked CV on a tiny single-session dataset: optimistic, no ride-level generalisation claim.**

**Data provenance: MIXED — REAL data + augmented/synthetic TRAINING windows; test folds real only | 4 rides, 297 windows (Smooth 194, Rough 60, Pothole 8, Irregularity 35)**

- Blocked CV: blocks of 15 s inside each ride (14 blocks; per class {'Irregularity': 4, 'Pothole': 3, 'Rough': 3, 'Smooth': 11}), StratifiedGroupKFold over blocks, k = 3, split seed 0 (first seed giving every class in every test fold), training windows within 2 s of a test window removed (purge).
  - fold 0: train 146 (purged 39), test 112, classes ['Irregularity', 'Pothole', 'Rough', 'Smooth']
  - fold 1: train 183 (purged 20), test 94, classes ['Irregularity', 'Pothole', 'Rough', 'Smooth']
  - fold 2: train 186 (purged 20), test 91, classes ['Irregularity', 'Pothole', 'Rough', 'Smooth']
- TRAINING folds extended with: augmentation=True, extra_train=None (extra windows per fold [186, 239, 242]); test folds contain real windows only.
- CV: blocked, 3 folds
- Windows: 1.0 s, step 0.5 s, resampled to 50 Hz; features: `invariant`, speed features: False
- Stationary windows (< 1.0 m/s) dropped: 1
- Pooled = metrics on all out-of-fold predictions; mean ± std = across folds (classes absent from a test fold are skipped for that fold).
- Picking the best of 3 models on the same CV is slightly optimistic.

## Overall

| Model | Accuracy (pooled) | Macro-F1 (pooled) | Macro-F1 (folds) |
|---|---|---|---|
| Rule baseline | 0.774 | 0.709 | 0.717 ± 0.108 |
| RandomForest | 0.886 | 0.755 | 0.752 ± 0.058 |
| HistGradientBoosting | 0.838 | 0.695 | 0.694 ± 0.044 |

## Per class (F1: pooled / folds mean ± std)

| Model | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Rule baseline | 0.844 / 0.843 ± 0.029 | 0.690 / 0.679 ± 0.039 | 0.636 / 0.648 ± 0.256 | 0.667 / 0.698 ± 0.152 |
| RandomForest | 0.939 / 0.938 ± 0.033 | 0.889 / 0.909 ± 0.061 | 0.476 / 0.456 ± 0.209 | 0.714 / 0.703 ± 0.031 |
| HistGradientBoosting | 0.901 / 0.899 ± 0.039 | 0.821 / 0.827 ± 0.115 | 0.375 / 0.378 ± 0.031 | 0.683 / 0.671 ± 0.034 |

## Per class precision / recall (pooled)

| Model | Smooth P / R | Rough P / R | Pothole P / R | Irregularity P / R |
|---|---|---|---|---|
| Rule baseline | 0.961 / 0.753 | 0.588 / 0.833 | 0.500 / 0.875 | 0.587 / 0.771 |
| RandomForest | 0.967 / 0.912 | 0.848 / 0.933 | 0.385 / 0.625 | 0.714 / 0.714 |
| HistGradientBoosting | 0.970 / 0.840 | 0.743 / 0.917 | 0.375 / 0.375 | 0.596 / 0.800 |

## Event level (tolerance ±1 s, evaluated distance 0.32 km)

Raw counts only (k detected / n labelled events; false alarms = count): the event numbers are far too small for rates.

> **Pothole has fewer than 10 labelled events (3).**

| Model | Pothole detected | Pothole false alarms | Irregularity detected | Irregularity false alarms |
|---|---|---|---|---|
| Rule baseline | 3/3 | 3 | 13/13 | 4 |
| RandomForest | 2/3 | 2 | 13/13 | 4 |
| HistGradientBoosting | 3/3 | 2 | 13/13 | 5 |

## Per ride (out-of-fold): windows / recall per class, macro-F1

**Rule baseline**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.623 | – | 8 / 0.875 | – | 0.851 |
| irregs | 69 / 0.797 | – | – | 35 / 0.771 | 0.824 |
| rug | – | 60 / 0.833 | – | – | 0.909 |
| smooth | 48 / 0.896 | – | – | – | 0.945 |

**RandomForest**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.818 | – | 8 / 0.625 | – | 0.807 |
| irregs | 69 / 0.957 | – | – | 35 / 0.714 | 0.885 |
| rug | – | 60 / 0.933 | – | – | 0.966 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

**HistGradientBoosting**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.805 | – | 8 / 0.375 | – | 0.696 |
| irregs | 69 / 0.942 | – | – | 35 / 0.800 | 0.906 |
| rug | – | 60 / 0.917 | – | – | 0.957 |
| smooth | 48 / 0.750 | – | – | – | 0.857 |

## Confusion matrices (pooled out-of-fold, rows = true, row-normalised)

**Rule baseline**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.75 (146) | 0.18 (34) | 0.00 (0) | 0.07 (14) |
| Rough | 0.10 (6) | 0.83 (50) | 0.00 (0) | 0.07 (4) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.88 (7) | 0.12 (1) |
| Irregularity | 0.00 (0) | 0.03 (1) | 0.20 (7) | 0.77 (27) |

**RandomForest**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.91 (177) | 0.05 (9) | 0.01 (1) | 0.04 (7) |
| Rough | 0.07 (4) | 0.93 (56) | 0.00 (0) | 0.00 (0) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.62 (5) | 0.38 (3) |
| Irregularity | 0.06 (2) | 0.03 (1) | 0.20 (7) | 0.71 (25) |

**HistGradientBoosting**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.84 (163) | 0.09 (17) | 0.01 (1) | 0.07 (13) |
| Rough | 0.07 (4) | 0.92 (55) | 0.00 (0) | 0.02 (1) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.38 (3) | 0.62 (5) |
| Irregularity | 0.03 (1) | 0.06 (2) | 0.11 (4) | 0.80 (28) |

## Windows per class

| Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|
| 194 | 60 | 8 | 35 |

_Data provenance: MIXED — REAL data + augmented/synthetic TRAINING windows; test folds real only | 4 rides, 297 windows (Smooth 194, Rough 60, Pothole 8, Irregularity 35)_
