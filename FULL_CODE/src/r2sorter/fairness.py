"""Fairness audit: how often are GENUINE customers of each group flagged (inspect or escalate)?"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .text import looks_non_english

SMOOTHING = 0.01  # +1 percentage point, like the official scorer


def _smoothed(flagged: pd.Series) -> float:
    return float(flagged.mean()) + SMOOTHING if len(flagged) else SMOOTHING


def _pair(flagged: pd.Series, in_group: pd.Series) -> dict[str, float]:
    first, second = _smoothed(flagged[in_group]), _smoothed(flagged[~in_group])
    ratio = max(first, second) / min(first, second)
    return {
        "group_rate": first - SMOOTHING,
        "other_rate": second - SMOOTHING,
        "ratio": ratio,
        "pair_score": float(min(1.0, max(0.0, 1 - (ratio - 1)))),
    }


def audit(frame: pd.DataFrame, decisions: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    """Frame needs label, customer_tenure_days, order_value_inr, customer_reason; decisions aligned by index."""
    genuine = frame["label"] != "abusive"
    flagged = (decisions != "auto_approve")[genuine]
    rows = frame[genuine]
    groups = {
        "new_vs_established": rows["customer_tenure_days"] < cfg["fairness"]["new_customer_days"],
        "low_value_vs_other": rows["order_value_inr"] < cfg["fairness"]["low_value_inr"],
        "non_english_vs_english": rows["customer_reason"].map(looks_non_english).astype(bool),
    }
    pairs = {name: _pair(flagged, mask) for name, mask in groups.items()}
    pairs["fairness_score"] = float(sum(p["pair_score"] for p in pairs.values()) / len(groups))
    return pairs
