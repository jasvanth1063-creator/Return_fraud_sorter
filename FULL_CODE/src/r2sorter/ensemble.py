"""Phase 7 helpers: weighted probability blending, simplex weight fitting and one routing evaluation per model.

The blend is P(y) = sum_m w_m * P_m(y) with w_m >= 0 and sum w_m = 1. Weights are fitted on the calibration part only
(never on the validation period or the hidden test), by minimising four-class log-loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, brier_score_loss

from .baselines import business_cost, expected_calibration_error
from .data import LABELS
from .fairness import audit
from .nlp import classification_report
from .router import cost_vectors, expected_costs, route
from .scoring import official_score

AB = LABELS.index("abusive")


def blend(members: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted average of probability matrices; rows are renormalised so they sum to exactly one."""
    weights = np.asarray(weights, dtype=float)
    if len(members) != len(weights):
        raise ValueError("one weight per member is required")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to one")
    mixed = sum(w * np.asarray(m, dtype=float) for w, m in zip(weights, members, strict=True))
    return mixed / mixed.sum(axis=1, keepdims=True)


def log_loss_of(probs: np.ndarray, y_index: np.ndarray) -> float:
    chosen = np.clip(probs[np.arange(len(y_index)), y_index], 1e-12, 1.0)
    return float(-np.log(chosen).mean())


def fit_simplex_weights(members: list[np.ndarray], y_index: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one that minimise four-class log-loss of the blend (softmax parametrisation)."""
    count = len(members)
    if count == 1:
        return np.ones(1)

    def objective(raw: np.ndarray) -> float:
        weights = np.exp(raw - raw.max())
        return log_loss_of(blend(members, weights / weights.sum()), y_index)

    best = None
    for start in (np.zeros(count), np.eye(count)[np.argmin([log_loss_of(m, y_index) for m in members])] * 2.0):
        result = minimize(
            objective, start, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000}
        )
        if best is None or result.fun < best.fun:
            best = result
    weights = np.exp(best.x - best.x.max())
    weights = np.where(weights / weights.sum() < 1e-3, 0.0, weights)  # drop members that contribute almost nothing
    return weights / weights.sum()


def evaluate_routing(
    name: str,
    probs_test: np.ndarray,
    p_abuse_cal: np.ndarray,
    truth_frame: pd.DataFrame,
    y_abuse: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Routing, official score, fairness and abusive-class metrics for one model's validation predictions."""
    decisions = route(p_abuse_cal, cfg)
    flagged = decisions != "auto_approve"
    report = classification_report(probs_test, truth_frame["label"])
    cost = business_cost(p_abuse_cal, y_abuse, cfg)
    submission = pd.DataFrame(
        {
            "return_id": truth_frame["return_id"].to_numpy(),
            "decision": decisions,
            "confidence": p_abuse_cal,
            "justification": "ensemble probability output without text explanation",
        }
    )
    official = official_score(submission, truth_frame)
    fairness = audit(truth_frame, pd.Series(decisions, index=truth_frame.index), cfg)
    raw = probs_test[:, AB]
    return {
        "model": name,
        "official_score": official["primary_score"],
        "cost_per_request": cost["cost_per_request"],
        "saving_vs_approve_all": cost["saving_vs_approve_all"],
        "abusive_PR_AUC": float(average_precision_score(y_abuse, raw)),
        "abusive_precision": float(report["per_class"].loc["abusive", "precision"]),
        "abusive_recall": float(report["per_class"].loc["abusive", "recall"]),
        "abusive_caught_by_routing": float(flagged[y_abuse == 1].mean()),
        "genuine_false_block_rate": float(flagged[y_abuse == 0].mean()),
        "genuine_escalated": cost["genuine_escalated"],
        "fairness_score": fairness["fairness_score"],
        "macro_f1": report["macro_f1"],
        "brier_calibrated": float(brier_score_loss(y_abuse, p_abuse_cal)),
        "ece_calibrated": float(expected_calibration_error(y_abuse, p_abuse_cal)),
        "escalate_rate": float((decisions == "escalate").mean()),
        "inspect_rate": float((decisions == "inspect").mean()),
    }


def row_costs(p_abuse: np.ndarray, y_abuse: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Official cost paid by every request when routed by minimum expected cost."""
    genuine, abusive = cost_vectors(cfg)
    chosen = expected_costs(p_abuse, cfg).argmin(axis=1)
    return np.where(np.asarray(y_abuse) == 1, abusive[chosen], genuine[chosen])


def paired_bootstrap(difference: np.ndarray, draws: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% interval of a per-request cost difference, resampling requests with replacement."""
    rng = np.random.default_rng(seed)
    difference = np.asarray(difference, dtype=float)
    means = np.empty(draws)
    for start in range(0, draws, 250):
        stop = min(draws, start + 250)
        picks = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        means[start:stop] = difference[picks].mean(axis=1)
    return float(difference.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def simplex_grid(count: int, step: float = 0.1) -> list[np.ndarray]:
    """Every weight vector on the simplex whose entries are multiples of `step`."""
    units = round(1 / step)
    grid: list[np.ndarray] = []

    def extend(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            grid.append(np.array([*prefix, remaining], dtype=float) / units)
            return
        for take in range(remaining + 1):
            extend([*prefix, take], remaining - take, slots - 1)

    extend([], units, count)
    return grid


def tune_weights_by_cost(
    members: list[np.ndarray],
    y_index: np.ndarray,
    y_abuse: np.ndarray,
    cfg: dict[str, Any],
    step: float = 0.1,
    tie: float = 5e-4,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Coarse grid over the simplex minimising the official routing cost of the blended raw P(abusive).

    Weights whose cost is within `tie` of the best are treated as tied and the lowest four-class log-loss wins.
    """
    rows = []
    for weights in simplex_grid(len(members), step):
        mixed = blend(members, weights)
        rows.append(
            {
                "weights": tuple(weights),
                "cost_per_request": float(row_costs(mixed[:, AB], y_abuse, cfg).mean()),
                "log_loss": log_loss_of(mixed, y_index),
            }
        )
    table = pd.DataFrame(rows)
    near = table[table["cost_per_request"] <= table["cost_per_request"].min() + tie]
    best = near.sort_values("log_loss").iloc[0]
    return np.array(best["weights"]), table.sort_values(["cost_per_request", "log_loss"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------------------------
# Fast official-score proxy (identical arithmetic to the official scorer's savings and fairness parts, no file I/O).
# primary = 0.7 * savings + 0.2 * fairness * min(1, savings / 0.2); the notebook checks it against the real scorer.
# --------------------------------------------------------------------------------------------------------------------

SMOOTHING = 0.01


def fairness_masks(frame: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Boolean group membership for every row: new customers, low-value orders, non-English text."""
    from .text import looks_non_english

    return {
        "new_vs_established": (frame["customer_tenure_days"] < cfg["fairness"]["new_customer_days"]).to_numpy(),
        "low_value_vs_other": (frame["order_value_inr"] < cfg["fairness"]["low_value_inr"]).to_numpy(),
        "non_english_vs_english": frame["customer_reason"].map(looks_non_english).astype(bool).to_numpy(),
    }


def fast_fairness(flagged: np.ndarray, y_abuse: np.ndarray, masks: dict[str, np.ndarray]) -> float:
    """Mean over groups of the pair score of the genuine customers' flag rates (same formula as the audit)."""
    genuine = np.asarray(y_abuse) == 0
    scores = []
    for mask in masks.values():
        inside, outside = genuine & mask, genuine & ~mask
        first = (flagged[inside].mean() if inside.any() else 0.0) + SMOOTHING
        second = (flagged[outside].mean() if outside.any() else 0.0) + SMOOTHING
        ratio = max(first, second) / min(first, second)
        scores.append(min(1.0, max(0.0, 1 - (ratio - 1))))
    return float(np.mean(scores))


def fast_official(
    p_abuse: np.ndarray, y_abuse: np.ndarray, masks: dict[str, np.ndarray], cfg: dict[str, Any]
) -> dict[str, float]:
    """Savings, fairness and primary score (size-chart part is zero in this project) for P(abusive) routed by cost."""
    genuine_cost, abusive_cost = cost_vectors(cfg)
    chosen = expected_costs(p_abuse, cfg).argmin(axis=1)
    y = np.asarray(y_abuse)
    cost = np.where(y == 1, abusive_cost[chosen], genuine_cost[chosen]).sum()
    baseline = np.where(y == 1, abusive_cost[0], genuine_cost[0]).sum()
    savings = float(min(1.0, max(-1.0, 1 - cost / baseline))) if baseline > 0 else 0.0
    fairness = fast_fairness(chosen != 0, y, masks)
    primary = 0.7 * savings + 0.2 * fairness * min(1.0, max(0.0, savings) / 0.2)
    return {
        "primary": float(primary),
        "savings": savings,
        "fairness": fairness,
        "genuine_escalated": int(((chosen == 2) & (y == 0)).sum()),
    }


def apply_group_multipliers(
    p_abuse: np.ndarray, masks: dict[str, np.ndarray], multipliers: dict[str, float]
) -> np.ndarray:
    """Scale P(abusive) by the product of the multipliers of every group the request belongs to (clipped to [0, 1])."""
    scale = np.ones(len(p_abuse))
    for name, factor in multipliers.items():
        scale = np.where(masks[name], scale * factor, scale)
    return np.clip(np.asarray(p_abuse, dtype=float) * scale, 0.0, 1.0)


def tune_group_multipliers(
    p_abuse: np.ndarray,
    y_abuse: np.ndarray,
    masks: dict[str, np.ndarray],
    cfg: dict[str, Any],
    grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25),
    rounds: int = 2,
) -> tuple[dict[str, float], float]:
    """Coordinate descent on the primary score; returns multipliers (1.0 = no change) and the score reached."""
    multipliers = {name: 1.0 for name in masks}
    best = fast_official(p_abuse, y_abuse, masks, cfg)["primary"]
    for _ in range(rounds):
        for name in masks:
            for value in grid:
                trial = {**multipliers, name: value}
                score = fast_official(apply_group_multipliers(p_abuse, masks, trial), y_abuse, masks, cfg)["primary"]
                if score > best + 1e-9:
                    best, multipliers = score, trial
    return multipliers, float(best)


def tune_weights_by_official(
    members: list[np.ndarray],
    y_abuse: np.ndarray,
    masks: dict[str, np.ndarray],
    cfg: dict[str, Any],
    method: str = "isotonic",
    step: float = 0.1,
    tie: float = 0.002,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Grid over the simplex maximising the primary score of cross-fit-calibrated blended P(abusive).

    Scores within `tie` of the best count as tied and the weights closest to equal weights win, which keeps the blend
    simple and stable when the data cannot tell nearby weightings apart.
    """
    from .conformal import cross_fit_calibrated

    equal = np.full(len(members), 1.0 / len(members))
    rows = []
    for weights in simplex_grid(len(members), step):
        raw = blend(members, weights)[:, AB]
        calibrated = cross_fit_calibrated(raw, np.asarray(y_abuse), method, folds=5)
        result = fast_official(calibrated, y_abuse, masks, cfg)
        rows.append({"weights": tuple(weights), **result, "distance_to_equal": float(np.linalg.norm(weights - equal))})
    table = pd.DataFrame(rows)
    near = table[table["primary"] >= table["primary"].max() - tie]
    best = near.sort_values("distance_to_equal").iloc[0]
    return np.array(best["weights"]), table.sort_values("primary", ascending=False).reset_index(drop=True)
