"""Probability calibration for P(abusive)."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-4


class AbuseCalibrator:
    """Maps raw model scores to honest probabilities.

    method is 'isotonic', 'sigmoid' (Platt scaling) or 'bagged_isotonic' (the average of isotonic fits on bootstrap
    resamples: a smoother, more stable step function for small calibration sets).
    """

    def __init__(self, method: str = "isotonic", n_bags: int = 50, seed: int = 0) -> None:
        if method not in {"isotonic", "sigmoid", "bagged_isotonic"}:
            raise ValueError("method must be 'isotonic', 'sigmoid' or 'bagged_isotonic'")
        self.method = method
        self.n_bags = n_bags
        self.seed = seed
        self._bag: list[IsotonicRegression] = []
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._platt = LogisticRegression(C=1e6, max_iter=1000)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        clipped = np.clip(p, _EPS, 1 - _EPS)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, truth: np.ndarray) -> AbuseCalibrator:
        """Fit on out-of-time predictions and their true 0/1 abuse labels."""
        if len(raw) != len(truth) or len(raw) == 0:
            raise ValueError("raw scores and truth must be non-empty and equally long")
        if self.method == "isotonic":
            self._iso.fit(raw, truth)
        elif self.method == "bagged_isotonic":
            raw, truth = np.asarray(raw, dtype=float), np.asarray(truth, dtype=float)
            rng = np.random.default_rng(self.seed)
            self._bag = []
            for _ in range(self.n_bags):
                pick = rng.integers(0, len(raw), len(raw))
                self._bag.append(
                    IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw[pick], truth[pick])
                )
        else:
            self._platt.fit(self._logit(raw), truth)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Calibrated probabilities, kept inside [0.001, 0.999] so none is claimed as certain."""
        raw = np.asarray(raw, dtype=float)
        if self.method == "isotonic":
            calibrated = self._iso.predict(raw)
        elif self.method == "bagged_isotonic":
            calibrated = np.mean([iso.predict(raw) for iso in self._bag], axis=0)
        else:
            calibrated = self._platt.predict_proba(self._logit(raw))[:, 1]
        return np.clip(calibrated, 0.001, 0.999)


def expected_calibration_error(truth: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """Average gap between confidence and observed rate over equal-width bins (same definition as the scorer)."""
    truth = np.asarray(truth, dtype=float)
    prob = np.asarray(prob, dtype=float)
    index = np.minimum((prob * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(truth[mask].mean() - prob[mask].mean())
    return float(total)
