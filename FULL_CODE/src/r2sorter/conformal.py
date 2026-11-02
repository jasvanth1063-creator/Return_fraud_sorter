"""Phase 8 helpers: four-class rescaling after calibration, cross-fitted calibration, split and Mondrian conformal sets.

Nonconformity score = 1 - P(true class). A class joins a prediction set when its score is at or below the threshold
learned on the calibration part. The marginal variant uses one threshold; the Mondrian variant uses one threshold per
true class, which gives a coverage guarantee for every class (under exchangeability of calibration and later data).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .calibration import AbuseCalibrator
from .data import LABELS

AB = LABELS.index("abusive")


def rescale_four_class(probs: np.ndarray, p_abuse_calibrated: np.ndarray) -> np.ndarray:
    """Replace the abusive column by the calibrated value and rescale the other classes so rows still sum to one."""
    probs = np.asarray(probs, dtype=float)
    out = probs.copy()
    others = [i for i in range(probs.shape[1]) if i != AB]
    remaining_old = np.clip(1.0 - probs[:, AB], 1e-9, None)
    out[:, AB] = p_abuse_calibrated
    out[:, others] = probs[:, others] / remaining_old[:, None] * (1.0 - p_abuse_calibrated)[:, None]
    return out / out.sum(axis=1, keepdims=True)


def cross_fit_calibrated(raw: np.ndarray, truth: np.ndarray, method: str, folds: int = 5) -> np.ndarray:
    """Calibrated P(abusive) for every row from a calibrator that never saw that row (contiguous time blocks)."""
    out = np.zeros(len(raw))
    blocks = np.array_split(np.arange(len(raw)), folds)
    for block in blocks:
        rest = np.setdiff1d(np.arange(len(raw)), block)
        out[block] = AbuseCalibrator(method).fit(raw[rest], truth[rest]).transform(raw[block])
    return out


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score (infinity if n is too small)."""
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def fit_thresholds(probs: np.ndarray, y_index: np.ndarray, alpha: float, mondrian: bool) -> np.ndarray:
    """One threshold per class (all equal for the marginal variant)."""
    scores = 1.0 - probs[np.arange(len(y_index)), y_index]
    if not mondrian:
        return np.full(probs.shape[1], conformal_quantile(scores, alpha))
    return np.array([conformal_quantile(scores[y_index == k], alpha) for k in range(probs.shape[1])])


def prediction_sets(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Boolean matrix: class k is in the set when 1 - P(k) <= threshold_k."""
    return (1.0 - probs) <= thresholds[None, :]


def ensure_nonempty(sets: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, float]:
    """Empty sets receive the most probable class (this can only raise coverage); returns the share that was empty."""
    empty = ~sets.any(axis=1)
    fixed = sets.copy()
    fixed[empty, probs[empty].argmax(axis=1)] = True
    return fixed, float(empty.mean())


def set_labels(sets: np.ndarray) -> list[str]:
    return ["{" + ", ".join(LABELS[k] for k in np.flatnonzero(row)) + "}" for row in sets]


def coverage(sets: np.ndarray, y_index: np.ndarray) -> float:
    return float(sets[np.arange(len(y_index)), y_index].mean())


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def recency_weights(dates: pd.Series, half_life_days: float) -> np.ndarray:
    """Weight 1 for the newest calibration row, halving every `half_life_days` into the past."""
    age = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    return 0.5 ** (age / half_life_days)


def weighted_conformal_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Recency-weighted split-conformal quantile (a new request counts as much as the heaviest calibration row)."""
    if len(scores) == 0:
        return float("inf")
    order = np.argsort(scores)
    ordered_scores, ordered_weights = np.asarray(scores)[order], np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(ordered_weights) / (ordered_weights.sum() + ordered_weights.max())
    position = int(np.searchsorted(cumulative, 1 - alpha, side="left"))
    return float("inf") if position >= len(ordered_scores) else float(ordered_scores[position])


def fit_thresholds_weighted(
    probs: np.ndarray, y_index: np.ndarray, weights: np.ndarray, alpha: float, mondrian: bool
) -> np.ndarray:
    """Like fit_thresholds but every calibration row carries a recency weight."""
    scores = 1.0 - probs[np.arange(len(y_index)), y_index]
    if not mondrian:
        return np.full(probs.shape[1], weighted_conformal_quantile(scores, weights, alpha))
    return np.array(
        [weighted_conformal_quantile(scores[y_index == k], weights[y_index == k], alpha) for k in range(probs.shape[1])]
    )
