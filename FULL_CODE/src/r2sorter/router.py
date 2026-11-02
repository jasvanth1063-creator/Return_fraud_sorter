"""Cost-sensitive three-way routing from a calibrated P(abusive)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DECISIONS = ("auto_approve", "inspect", "escalate")


def cost_vectors(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(cost if genuine, cost if abusive) for each decision, in DECISIONS order."""
    costs = cfg["costs"]
    genuine = np.array([costs["genuine"][d] for d in DECISIONS], dtype=float)
    abusive = np.array([costs["abusive"][d] for d in DECISIONS], dtype=float)
    return genuine, abusive


def expected_costs(p_abuse: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Expected cost of each decision for every request, shape (n, 3)."""
    genuine, abusive = cost_vectors(cfg)
    p = np.asarray(p_abuse, dtype=float).reshape(-1, 1)
    return (1 - p) * genuine + p * abusive


def route(p_abuse: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Pick the decision with the lowest expected cost."""
    return np.array(DECISIONS)[expected_costs(p_abuse, cfg).argmin(axis=1)]


def break_even_thresholds(cfg: dict[str, Any]) -> dict[str, float]:
    """P(abusive) at which the cheapest decision changes."""
    genuine, abusive = cost_vectors(cfg)

    def crossing(first: int, second: int) -> float:
        slope_g = genuine[first] - genuine[second]
        slope_a = abusive[first] - abusive[second]
        return float(slope_g / (slope_g - slope_a))

    return {
        "approve_to_inspect": crossing(0, 1),
        "inspect_to_escalate": crossing(1, 2),
    }


def decision_table(p_abuse: np.ndarray, cfg: dict[str, Any]) -> pd.DataFrame:
    """Decision plus the expected cost of each option, for audit."""
    costs = expected_costs(p_abuse, cfg)
    table = pd.DataFrame(costs, columns=[f"exp_cost_{d}" for d in DECISIONS])
    table.insert(0, "decision", np.array(DECISIONS)[costs.argmin(axis=1)])
    return table
