# Smart Cycle — classifier evaluation

> **blocked CV on a tiny single-session dataset: optimistic, no ride-level generalisation claim.**

**Data provenance: REAL | 4 rides, 297 windows (Smooth 194, Rough 60, Pothole 8, Irregularity 35)**

- Blocked CV: blocks of 15 s inside each ride (14 blocks; per class {'Irregularity': 4, 'Pothole': 3, 'Rough': 3, 'Smooth': 11}), StratifiedGroupKFold over blocks, k = 3, split seed 0 (first seed giving every class in every test fold), training windows within 2 s of a test window removed (purge).
  - fold 0: train 146 (purged 39), test 112, classes ['Irregularity', 'Pothole', 'Rough', 'Smooth']
  - fold 1: train 183 (purged 20), test 94, classes ['Irregularity', 'Pothole', 'Rough', 'Smooth']
  - fold 2: train 186 (purged 20), test 91, classes ['Irregularity', 'Pothole', 'Rough', 'Smooth']
- CV: blocked, 3 folds
- Windows: 1.0 s, step 0.5 s, resampled to 50 Hz; features: `invariant`, speed features: False
- Stationary windows (< 1.0 m/s) dropped: 1
- Pooled = metrics on all out-of-fold predictions; mean ± std = across folds (classes absent from a test fold are skipped for that fold).
- Picking the best of 3 models on the same CV is slightly optimistic.

## Overall

| Model | Accuracy (pooled) | Macro-F1 (pooled) | Macro-F1 (folds) |
|---|---|---|---|
| Rule baseline | 0.801 | 0.739 | 0.739 ± 0.067 |
| RandomForest | 0.879 | 0.778 | 0.790 ± 0.063 |
| HistGradientBoosting | 0.832 | 0.607 | 0.606 ± 0.034 |

## Per class (F1: pooled / folds mean ± std)

| Model | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Rule baseline | 0.866 / 0.864 ± 0.087 | 0.683 / 0.720 ± 0.124 | 0.700 / 0.669 ± 0.195 | 0.707 / 0.704 ± 0.095 |
| RandomForest | 0.925 / 0.924 ± 0.063 | 0.852 / 0.890 ± 0.132 | 0.588 / 0.600 ± 0.196 | 0.747 / 0.745 ± 0.033 |
| HistGradientBoosting | 0.925 / 0.926 ± 0.057 | 0.788 / 0.826 ± 0.176 | 0.111 / 0.074 ± 0.105 | 0.602 / 0.598 ± 0.057 |

## Per class precision / recall (pooled)

| Model | Smooth P / R | Rough P / R | Pothole P / R | Irregularity P / R |
|---|---|---|---|---|
| Rule baseline | 0.904 / 0.830 | 0.683 / 0.683 | 0.583 / 0.875 | 0.617 / 0.829 |
| RandomForest | 0.927 / 0.923 | 0.891 / 0.817 | 0.556 / 0.625 | 0.700 / 0.800 |
| HistGradientBoosting | 0.923 / 0.928 | 0.932 / 0.683 | 0.100 / 0.125 | 0.521 / 0.714 |

## Event level (tolerance ±1 s, evaluated distance 0.32 km)

Raw counts only (k detected / n labelled events; n predicted event units; false alarms = unmatched units): the event numbers are far too small for rates. Matching: one-to-one: a labelled event is matched by at most one predicted event unit of the same class whose peak lies within the tolerance of the labelled interval; merged predicted events are cut into units of at most 2 s, each unmatched unit is a false alarm.

> **Pothole has fewer than 10 labelled events (3).**

| Model | Pothole detected | Pothole predicted | Pothole false alarms | Irregularity detected | Irregularity predicted | Irregularity false alarms |
|---|---|---|---|---|---|---|
| Rule baseline | 3/3 | 5 | 2 | 11/13 | 17 | 6 |
| RandomForest | 3/3 | 4 | 1 | 11/13 | 16 | 5 |
| HistGradientBoosting | 2/3 | 9 | 7 | 12/13 | 21 | 9 |
| _reference: whole-ride Pothole_ | 3/3 | 75 | 72 | 0/13 | 0 | 0 |
| _reference: whole-ride Irregularity_ | 0/3 | 0 | 0 | 13/13 | 75 | 62 |
| _reference: all Smooth_ | 0/3 | 0 | 0 | 0/13 | 0 | 0 |

Reference rows: degenerate predictors (every evaluated window predicted as that class) scored with the same metric, to calibrate expectations.

## Per ride (out-of-fold): windows / recall per class, macro-F1

**Rule baseline**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.727 | – | 8 / 0.875 | – | 0.888 |
| irregs | 69 / 0.826 | – | – | 35 / 0.829 | 0.850 |
| rug | – | 60 / 0.683 | – | – | 0.812 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

**RandomForest**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.844 | – | 8 / 0.625 | – | 0.815 |
| irregs | 69 / 0.957 | – | – | 35 / 0.800 | 0.903 |
| rug | – | 60 / 0.817 | – | – | 0.899 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

**HistGradientBoosting**

| Ride | Smooth n / R | Rough n / R | Pothole n / R | Irregularity n / R | Macro-F1 |
|---|---|---|---|---|---|
| bump | 77 / 0.870 | – | 8 / 0.125 | – | 0.556 |
| irregs | 69 / 0.942 | – | – | 35 / 0.714 | 0.862 |
| rug | – | 60 / 0.683 | – | – | 0.812 |
| smooth | 48 / 1.000 | – | – | – | 1.000 |

## Confusion matrices (pooled out-of-fold, rows = true, row-normalised)

**Rule baseline**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.83 (161) | 0.09 (18) | 0.00 (0) | 0.08 (15) |
| Rough | 0.28 (17) | 0.68 (41) | 0.00 (0) | 0.03 (2) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.88 (7) | 0.12 (1) |
| Irregularity | 0.00 (0) | 0.03 (1) | 0.14 (5) | 0.83 (29) |

**RandomForest**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.92 (179) | 0.03 (5) | 0.01 (1) | 0.05 (9) |
| Rough | 0.18 (11) | 0.82 (49) | 0.00 (0) | 0.00 (0) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.62 (5) | 0.38 (3) |
| Irregularity | 0.09 (3) | 0.03 (1) | 0.09 (3) | 0.80 (28) |

**HistGradientBoosting**

| true \ pred | Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|---|
| Smooth | 0.93 (180) | 0.02 (3) | 0.01 (2) | 0.05 (9) |
| Rough | 0.18 (11) | 0.68 (41) | 0.02 (1) | 0.12 (7) |
| Pothole | 0.00 (0) | 0.00 (0) | 0.12 (1) | 0.88 (7) |
| Irregularity | 0.11 (4) | 0.00 (0) | 0.17 (6) | 0.71 (25) |

## Experiments / ablations (RandomForest, same CV protocol: blocked)

| Experiment | Windows | Macro-F1 (pooled) | Macro-F1 (folds) | F1 Smooth | F1 Rough | F1 Pothole | F1 Irregularity | Pothole events | Irregularity events |
|---|---|---|---|---|---|---|---|---|---|
| baseline (invariant, 1 s, no speed) | 297 | 0.778 | 0.790 ± 0.063 | 0.925 | 0.852 | 0.588 | 0.747 | 3/3 (1 FA) | 11/13 (5 FA) |
| (i) WITH speed features | 297 | 0.787 | 0.795 ± 0.054 | 0.933 | 0.872 | 0.588 | 0.757 | 3/3 (1 FA) | 11/13 (5 FA) |
| (i) speed features ONLY (confound diagnostic) | 297 | 0.342 | 0.327 ± 0.046 | 0.633 | 0.492 | 0.000 | 0.242 | 0/3 (8 FA) | 12/13 (17 FA) |
| (ii) invariant+fixed | 297 | 0.786 | 0.791 ± 0.052 | 0.927 | 0.872 | 0.588 | 0.757 | 3/3 (1 FA) | 11/13 (4 FA) |
| (iii) 2 s windows | 297 | 0.788 | 0.781 ± 0.066 | 0.920 | 0.784 | 0.727 | 0.720 | 3/3 (3 FA) | 12/13 (14 FA) |
| DIAGNOSTIC 3 classes (Smooth/Rough/Impact=Pothole+Irregularity) | 297 | 0.885 | 0.895 ± 0.050 | – | – | – | – | – | – |
| (iv) train handlebar -> test pocket (invariant); ONLY Pothole labelled, no overall score | 2 | – | – | – | – | – | – | 0/1 | – |
| (iv) train handlebar -> test pocket (invariant+fixed); ONLY Pothole labelled, no overall score | 2 | – | – | – | – | – | – | 0/1 | – |

### Speed per class (moving windows, m/s)

| Class | n | mean | median | std |
|---|---|---|---|---|
| Smooth | 194 | 2.20 | 2.11 | 0.41 |
| Rough | 60 | 1.67 | 1.62 | 0.24 |
| Pothole | 8 | 2.68 | 2.69 | 0.20 |
| Irregularity | 35 | 2.49 | 2.49 | 0.31 |

### DIAGNOSTIC: 3 classes (Smooth / Rough / Impact = Pothole + Irregularity)

_Diagnostic only: Pothole and Irregularity merged into Impact. The product and the final model keep the 4 plan classes._

Macro-F1 pooled 0.885 (folds 0.895 ± 0.050); Impact events detected 15/16, false alarms 2.

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| Smooth | 0.937 | 0.923 | 0.930 | 194 |
| Rough | 0.864 | 0.850 | 0.857 | 60 |
| Impact | 0.830 | 0.907 | 0.867 | 43 |

| true \ pred | Smooth | Rough | Impact |
|---|---|---|---|
| Smooth | 179 | 7 | 8 |
| Rough | 9 | 51 | 0 |
| Impact | 3 | 1 | 39 |

### (iv) Orientation test, train handlebar -> test pocket (`invariant`)

**Only meaningful for the classes labelled in pocket: Pothole. No overall score is reported.**

| Class | Windows correct | Predicted as | Events detected |
|---|---|---|---|
| Pothole | 0/2 | {'Smooth': 0, 'Rough': 0, 'Pothole': 0, 'Irregularity': 2} | 0/1 |

### (iv) Orientation test, train handlebar -> test pocket (`invariant+fixed`)

**Only meaningful for the classes labelled in pocket: Pothole. No overall score is reported.**

| Class | Windows correct | Predicted as | Events detected |
|---|---|---|---|
| Pothole | 0/2 | {'Smooth': 0, 'Rough': 0, 'Pothole': 0, 'Irregularity': 2} | 0/1 |

> **Pothole has fewer than 10 labelled events (3): every Pothole result is anecdotal; counts are reported as k/n, never as rates.**

> **CONFOUNDING WARNING: speed alone gives macro-F1 0.342 (chance ~0.25) and class mean speeds differ by x1.60: part of what the model learns may be 'which recording / which speed', not the road. Compare with the no-speed row.**

## Windows per class

| Smooth | Rough | Pothole | Irregularity |
|---|---|---|---|
| 194 | 60 | 8 | 35 |

_Data provenance: REAL | 4 rides, 297 windows (Smooth 194, Rough 60, Pothole 8, Irregularity 35)_
