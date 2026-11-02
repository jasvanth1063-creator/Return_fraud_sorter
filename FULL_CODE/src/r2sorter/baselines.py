"""Phase 5: baseline classifiers, probability-quality metrics and the official-cost routing evaluation.

Hyper-parameters are fixed in advance (not tuned on the validation period). Every model returns four-class
probabilities in ``data.LABELS`` order; P(abusive) is the column ``ABUSIVE``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.tree import DecisionTreeClassifier

from .data import LABELS
from .router import DECISIONS, cost_vectors, route

ABUSIVE = LABELS.index("abusive")
MODEL_ORDER = (
    "dummy",
    "logreg",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "hist_gb",
    "catboost",
)  # simplest first
# which feature matrix each model reads: "linear" = imputed + scaled + one-hot, "tree" = raw numerics + typed categories
MATRIX_KIND = {
    "dummy": "linear",
    "logreg": "linear",
    "decision_tree": "linear",
    "random_forest": "linear",
    "extra_trees": "linear",
    "hist_gb": "tree",
    "catboost": "tree",
}
STOCHASTIC = {"random_forest", "extra_trees", "hist_gb", "catboost"}


def catboost_inputs(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """CatBoost wants categorical columns as strings; returns (frame, names of categorical columns)."""
    converted = features.copy()
    names = [c for c in converted.columns if isinstance(converted[c].dtype, pd.CategoricalDtype)]
    for column in names:
        converted[column] = converted[column].astype(str)
    return converted, names


def build_model(name: str, seed: int, cat_features: list[str] | None = None) -> Any:
    """Fresh, untuned estimator with the documented fixed settings."""
    factories: dict[str, Callable[[], Any]] = {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "logreg": lambda: LogisticRegression(C=1.0, max_iter=3000),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, random_state=seed),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=seed
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=100, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=seed
        ),
        "hist_gb": lambda: HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=5.0,
            categorical_features="from_dtype",
            early_stopping=False,
            random_state=seed,
        ),
    }
    if name == "catboost":
        from catboost import CatBoostClassifier  # imported lazily: only this baseline needs it

        return CatBoostClassifier(
            iterations=200,
            learning_rate=0.08,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="MultiClass",
            random_seed=seed,
            thread_count=2,
            verbose=0,
            cat_features=cat_features or [],
            allow_writing_files=False,
        )
    if name not in factories:
        raise KeyError(f"unknown baseline {name!r}")
    return factories[name]()


def predict_full(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Four-column probabilities in LABELS order even if a class were missing from the training rows."""
    probs = model.predict_proba(features)
    full = np.zeros((len(features), len(LABELS)))
    full[:, np.asarray(model.classes_, dtype=int)] = probs
    return full


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Equal-mass-bin ECE: weighted gap between mean predicted and observed rate."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(p) * abs(p[c].mean() - y[c].mean()) for c in chunks if len(c)))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Per equal-mass bin: rows, mean predicted P(abusive), observed abusive rate."""
    chunks = np.array_split(np.argsort(p), bins)
    return pd.DataFrame(
        {
            "rows": [len(c) for c in chunks],
            "mean_predicted": [p[c].mean() for c in chunks],
            "observed_rate": [y[c].mean() for c in chunks],
        }
    ).round(4)


def abuse_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Probability quality for P(abusive): log loss, Brier, average precision, ECE."""
    clipped = np.clip(p, 1e-4, 1 - 1e-4)
    return {
        "abuse_log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "abuse_brier": float(brier_score_loss(y, clipped)),
        "abuse_AP": float(average_precision_score(y, p)),
        "abuse_ECE": expected_calibration_error(y, clipped),
    }


def business_cost(p: np.ndarray, y: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    """Route by expected cost under the OFFICIAL matrix and measure what that routing costs on true labels."""
    decisions = route(p, cfg)
    index = np.array([DECISIONS.index(d) for d in decisions])
    genuine, abusive = cost_vectors(cfg)
    cost = np.where(y == 1, abusive[index], genuine[index]).mean()
    approve_all = np.where(y == 1, abusive[0], genuine[0]).mean()
    return {
        "cost_per_request": float(cost),
        "saving_vs_approve_all": float(1 - cost / approve_all),
        "genuine_escalated": int(((y == 0) & (decisions == "escalate")).sum()),
        "abusive_approved": int(((y == 1) & (decisions == "auto_approve")).sum()),
        "inspect_share": float((decisions == "inspect").mean()),
        "escalate_share": float((decisions == "escalate").mean()),
    }
