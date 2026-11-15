"""The three models behind one interface (fit / predict / predict_proba / classes_).

All are sklearn Pipelines: median imputation (fitted on the training fold only) + classifier.
Hyperparameters are FIXED (no tuning anywhere).
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from smartcycle.models.rule import RuleBaseline

MODEL_NAMES = ("rule", "rf", "hgb")
MODEL_LABELS = {"rule": "Rule baseline", "rf": "RandomForest", "hgb": "HistGradientBoosting"}


class BalancedHGB(HistGradientBoostingClassifier):
    """HistGradientBoosting with balanced sample weights computed at fit time (does not rely
    on a `class_weight` argument being available in the installed sklearn)."""

    def fit(self, X, y, sample_weight=None):  # noqa: D102
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight)


def make_model(name: str) -> Pipeline:
    if name == "rule":
        clf = RuleBaseline()
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     class_weight="balanced_subsample", random_state=0, n_jobs=-1)
    elif name == "hgb":
        clf = BalancedHGB(max_iter=200, learning_rate=0.1, early_stopping=False, random_state=0)
    else:
        raise ValueError(f"Unknown model {name!r}; use one of {MODEL_NAMES}")
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    return Pipeline([("impute", imp), ("clf", clf)]).set_output(transform="pandas")
