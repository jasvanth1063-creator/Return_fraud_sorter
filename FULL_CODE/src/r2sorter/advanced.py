"""Phase 6 helpers: robust LightGBM variants, efficiency measurement and importance tables.

TabPFN and XGBoost are intentionally not imported here: they are not installed in the project environment, and TabPFN
cannot fetch its weights in the offline workspace (the notebook records that as a limitation instead of forcing a run).
"""

from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

from .data import LABELS

ABUSIVE = LABELS.index("abusive")


def class_weights(y_index: np.ndarray, power: float = 0.5) -> dict[int, float]:
    """Mild inverse-frequency weights: (max_share / share) ** power, so the largest class has weight 1."""
    shares = np.bincount(y_index, minlength=len(LABELS)) / len(y_index)
    return {i: float((shares.max() / max(share, 1e-9)) ** power) for i, share in enumerate(shares)}


def lgbm_params(variant: str, seed: int) -> dict[str, Any]:
    """Fixed (untuned) configurations. 'plain' regularised; 'weighted' adds mild class weights; 'robust' adds
    stronger regularisation, randomised splits and smoothing, which damp the effect of noisy or ambiguous labels."""
    base: dict[str, Any] = {
        "objective": "multiclass",
        "num_class": len(LABELS),
        "n_estimators": 150,
        "learning_rate": 0.08,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "reg_lambda": 8.0,
        "random_state": seed,
        "n_jobs": 2,
        "verbose": -1,
    }
    if variant == "plain":
        return base
    if variant == "weighted":
        return base
    if variant == "robust":
        return {
            **base,
            "min_child_samples": 100,
            "reg_lambda": 25.0,
            "extra_trees": True,
            "path_smooth": 20.0,
            "num_leaves": 24,
        }
    raise KeyError(f"unknown LightGBM variant {variant!r}")


def fit_lgbm(variant: str, seed: int, features: pd.DataFrame, y_index: np.ndarray) -> lgb.LGBMClassifier:
    """Fit one variant; 'weighted' and 'robust' use mild class weights, 'plain' does not."""
    model = lgb.LGBMClassifier(**lgbm_params(variant, seed))
    weights = None
    if variant in {"weighted", "robust"}:
        table = class_weights(y_index)
        weights = np.array([table[i] for i in y_index])
    return model.fit(features, y_index, sample_weight=weights)


def measure(fit: Any, predict: Any, rows_for_inference: int) -> tuple[Any, dict[str, float]]:
    """Run ``fit()`` then ``predict()``; return (prediction, efficiency). Memory is RSS growth (MB)."""
    process = psutil.Process()
    before = process.memory_info().rss
    started = time.perf_counter()
    model = fit()
    train_seconds = time.perf_counter() - started
    after_fit = process.memory_info().rss
    started = time.perf_counter()
    output = predict(model)
    infer = time.perf_counter() - started
    return output, {
        "train_seconds": round(train_seconds, 2),
        "inference_ms_per_1k_rows": round(infer * 1000 / max(rows_for_inference / 1000, 1e-9), 2),
        "rss_growth_mb": round(max(after_fit - before, 0) / 1e6, 1),
    }


def importance_table(model: lgb.LGBMClassifier, columns: list[str], top: int = 15) -> pd.DataFrame:
    """Gain-based importance, normalised to sum to 1, largest first."""
    gain = model.booster_.feature_importance(importance_type="gain")
    table = pd.Series(gain / max(gain.sum(), 1e-12), index=columns, name="gain_share").sort_values(ascending=False)
    return table.head(top).round(4).to_frame()
