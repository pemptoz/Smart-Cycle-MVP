# Smart Cycle — classifier evaluation

**Data provenance: SYNTHETIC — SYNTHETIC DATA, NOT REAL RESULTS | 12 rides, 7901 windows (Smooth 4692, Rough 2137, Pothole 544, Irregularity 528)**

- CV: grouped, 5 folds (grouped by ride: no ride in both train and test)
- Windows: 1.0 s, step 0.5 s, resampled to 50 Hz; features: `invariant`, speed features: True
- Stationary windows (< 1.0 m/s) dropped: 271
- Pooled = metrics on all out-of-fold predictions; mean ± std = across folds (classes absent from a test fold are skipped for that fold).
- Picking the best of 3 models on the same CV is slightly optimistic.

## Overall

| Model | Accuracy (pooled) | Macro-F1 (pooled) | Macro-F1 (folds) |
|---|---|---|---|
| Rule baseline | 0.786 | 0.690 | 0.701 ± 0.035 |
| RandomForest | 0.934 | 0.883 | 0.883 ± 0.016 |
| HistGradientBoosting | 0.932 | 0.881 | 0.881 ± 0.024 |

## Per class (F1: pooled / folds mean ± std)

| Model | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Rule baseline | 0.867 / 0.864 ± 0.057 | 0.689 / 0.694 ± 0.061 | 0.739 / 0.741 ± 0.029 | 0.464 / 0.505 ± 0.106 |
| RandomForest | 0.958 / 0.959 ± 0.015 | 0.932 / 0.934 ± 0.031 | 0.818 / 0.817 ± 0.023 | 0.824 / 0.821 ± 0.030 |
| HistGradientBoosting | 0.955 / 0.956 ± 0.021 | 0.931 / 0.934 ± 0.037 | 0.819 / 0.817 ± 0.029 | 0.820 / 0.819 ± 0.034 |

## Per class precision / recall (pooled)

| Model | Smooth P / R | Rough P / R | Pothole P / R | Irregularity P / R |
|---|---|---|---|---|
| Rule baseline | 0.850 / 0.885 | 0.684 / 0.694 | 0.895 / 0.629 | 0.494 / 0.438 |
| RandomForest | 0.944 / 0.973 | 0.912 / 0.953 | 0.956 / 0.715 | 0.927 / 0.742 |
| HistGradientBoosting | 0.948 / 0.962 | 0.909 / 0.954 | 0.934 / 0.730 | 0.869 / 0.777 |

## Event level (tolerance ±1 s, evaluated distance 18.07 km)

| Model | Pothole recall | Pothole false alarms/km | Irregularity recall | Irregularity false alarms/km |
|---|---|---|---|---|
| Rule baseline | 0.947 (161/170) | 0.39 | 0.586 (89/152) | 0.66 |
| RandomForest | 0.988 (168/170) | 0.66 | 0.974 (148/152) | 0.94 |
| HistGradientBoosting | 0.988 (168/170) | 1.11 | 0.980 (149/152) | 1.49 |

## Confusion matrices (pooled out-of-fold, rows = true, row-normalised)

**Rule baseline**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.88 (4151) | 0.09 (430) | 0.00 (1) | 0.02 (110) |
| Rough | 0.24 (504) | 0.69 (1484) | 0.02 (37) | 0.05 (112) |
| Pothole | 0.17 (94) | 0.17 (93) | 0.63 (342) | 0.03 (15) |
| Irregularity | 0.25 (133) | 0.31 (162) | 0.00 (2) | 0.44 (231) |

**RandomForest**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.97 (4566) | 0.02 (89) | 0.00 (12) | 0.01 (25) |
| Rough | 0.04 (91) | 0.95 (2036) | 0.00 (6) | 0.00 (4) |
| Pothole | 0.17 (95) | 0.11 (58) | 0.72 (389) | 0.00 (2) |
| Irregularity | 0.16 (87) | 0.09 (49) | 0.00 (0) | 0.74 (392) |

**HistGradientBoosting**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.96 (4515) | 0.02 (104) | 0.00 (20) | 0.01 (53) |
| Rough | 0.04 (83) | 0.95 (2038) | 0.00 (8) | 0.00 (8) |
| Pothole | 0.17 (91) | 0.10 (55) | 0.73 (397) | 0.00 (1) |
| Irregularity | 0.14 (73) | 0.09 (45) | 0.00 (0) | 0.78 (410) |

## Ablations (RandomForest, ride-grouped CV)

| Ablation | Windows | Macro-F1 (pooled) | Macro-F1 (folds) | Pothole ev. recall | Irregularity ev. recall |
|---|---|---|---|---|---|
| baseline (invariant, 1 s, speed) | 7901 | 0.883 | 0.883 ± 0.016 | 0.988 | 0.974 |
| + fixed axes | 7901 | 0.885 | 0.885 ± 0.018 | 0.994 | 0.967 |
| no speed features | 7901 | 0.882 | 0.882 ± 0.016 | 0.988 | 0.974 |
| 2 s windows | 7876 | 0.900 | 0.901 ± 0.025 | 0.994 | 0.974 |
| mount=handlebar (subset of baseline OOF) | 4045 | 0.906 | – | – | – |
| mount=pocket (subset of baseline OOF) | 3856 | 0.858 | – | – | – |

## Windows per class

| Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|
| 4692 | 2137 | 544 | 528 |

_Data provenance: SYNTHETIC — SYNTHETIC DATA, NOT REAL RESULTS | 12 rides, 7901 windows (Smooth 4692, Rough 2137, Pothole 544, Irregularity 528)_
