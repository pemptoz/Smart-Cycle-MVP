"""Rule-based baseline: hand-structured decision list, thresholds fitted on training data only.

    peak |a_vert| > T_pothole        -> Pothole
    elif peak a_hmag > T_irreg       -> Irregularity
    elif rms a_vert > T_rough        -> Rough
    else                             -> Smooth

Thresholds come from a small grid search on the TRAINING fold only (candidate thresholds =
quantiles of each feature; coordinate ascent: each threshold is optimised in turn with the
others fixed, a few rounds) maximising macro-F1 on that same training fold. Nothing is tuned
on test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from smartcycle.io.labels import LABELS

QUANTILES = tuple(np.round(np.linspace(0.30, 0.995, 100), 4))
ROUNDS = 3


def macro_f1_codes(y: np.ndarray, p: np.ndarray, k: int) -> float:
    """Fast macro-F1 for integer-coded labels 0..k-1 (zero_division = 0)."""
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


class RuleBaseline(ClassifierMixin, BaseEstimator):
    def __init__(self, quantiles: tuple[float, ...] = QUANTILES):
        self.quantiles = quantiles

    @staticmethod
    def _scores(X: pd.DataFrame) -> np.ndarray:
        peak_v = np.maximum(np.abs(X["a_vert__min"]), np.abs(X["a_vert__max"]))
        return np.column_stack([peak_v, X["a_hmag__max"], X["a_vert__rms"]]).astype(float)

    def _decide(self, S: np.ndarray, th: tuple[float, float, float]) -> np.ndarray:
        """Integer codes in self.classes_ order."""
        code = {c: i for i, c in enumerate(self.classes_)}
        out = np.full(len(S), code.get("Smooth", 0))
        for col, cls in ((2, "Rough"), (1, "Irregularity"), (0, "Pothole")):  # last wins
            if cls in code:
                out = np.where(S[:, col] > th[col], code[cls], out)
        return out

    def fit(self, X: pd.DataFrame, y) -> "RuleBaseline":
        y = np.asarray(y).astype(str)
        self.classes_ = np.array([c for c in LABELS if c in set(y)])
        code = {c: i for i, c in enumerate(self.classes_)}
        yc = np.array([code[v] for v in y])
        S = self._scores(X)
        grids = [np.unique(np.quantile(S[:, j], self.quantiles)) for j in range(3)]
        th = [float(np.median(g)) for g in grids]
        best = macro_f1_codes(yc, self._decide(S, tuple(th)), len(self.classes_))
        for _ in range(ROUNDS):
            for j in (0, 1, 2):
                for v in np.r_[grids[j], np.inf]:  # inf = rule disabled
                    cand = list(th)
                    cand[j] = float(v)
                    f = macro_f1_codes(yc, self._decide(S, tuple(cand)), len(self.classes_))
                    if f > best:
                        best, th = f, cand
        best_th = tuple(th)
        self.thresholds_ = {"pothole_peak_vert": float(best_th[0]),
                            "irregularity_peak_hmag": float(best_th[1]),
                            "rough_rms_vert": float(best_th[2])}
        self._th = best_th
        self.train_macro_f1_ = best
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self._decide(self._scores(X), self._th)]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        codes = self._decide(self._scores(X), self._th)
        P = np.zeros((len(codes), len(self.classes_)))
        P[np.arange(len(codes)), codes] = 1.0
        return P
