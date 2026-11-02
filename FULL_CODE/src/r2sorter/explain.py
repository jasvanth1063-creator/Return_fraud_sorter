"""Plain-language justifications built from the real feature values that drove each prediction."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Phrase = Callable[[pd.Series], str | None]


def _days(row: pd.Series) -> str | None:
    value = row["days_since_delivery"]
    return f"returned {int(value)} days after delivery" if value >= 10 else None


def _cod(row: pd.Series) -> str | None:
    return "cash-on-delivery order" if row["cod_flag"] == 1 else None


def _prior_rate(row: pd.Series) -> str | None:
    rate = row["prior_return_rate"]
    return f"{rate:.0%} of past orders returned" if pd.notna(rate) and rate >= 0.3 else None


def _recent(row: pd.Series) -> str | None:
    n = row.get("hist_returns_30d", 0)
    return f"{int(n)} earlier returns in the last 30 days" if n >= 1 else None


def _repeat_text(row: pd.Series) -> str | None:
    if row["hist_same_text"] >= 1 or row["hist_max_text_sim"] >= 0.8:
        return "reason text repeats the customer's earlier returns"
    return None


def _past_abuse(row: pd.Series) -> str | None:
    n = row.get("lab_n_abusive", 0)
    return f"{int(n)} earlier return(s) already judged abusive" if n >= 1 else None


def _single_item(row: pd.Series) -> str | None:
    return "single-item order" if row["items_in_order"] == 1 else None


def _price(row: pd.Series) -> str | None:
    return f"high-value item (Rs {int(row['item_price_inr'])})" if row["item_price_inr"] >= 3000 else None


def _flag(column: str, text: str) -> Phrase:
    return lambda row: text if row.get(column, 0) >= 1 else None


def _text_genuine(row: pd.Series) -> str | None:
    best = max(("fit", "quality", "wrong_item"), key=lambda k: row[_TEXT_COLUMNS[k]])
    share = row[_TEXT_COLUMNS[best]]
    return (
        f"reason text reads like a genuine {best.replace('_', ' ')} complaint ({share:.0%})" if share >= 0.6 else None
    )


_TEXT_COLUMNS = {
    "fit": "txt_p_genuine_fit_issue",
    "quality": "txt_p_genuine_quality_issue",
    "wrong_item": "txt_p_genuine_wrong_item",
}

PHRASES: dict[str, Phrase] = {
    "days_since_delivery": _days,
    "cod_flag": _cod,
    "payment_method": _cod,
    "prior_return_rate": _prior_rate,
    "hist_rate_gap": _prior_rate,
    "hist_returns_30d": _recent,
    "hist_returns_60d": _recent,
    "hist_returns_90d": _recent,
    "hist_same_text": _repeat_text,
    "hist_max_text_sim": _repeat_text,
    "hist_n_similar_text": _repeat_text,
    "lab_n_abusive": _past_abuse,
    "lab_abuse_share": _past_abuse,
    "lab_days_since_abusive": _past_abuse,
    "items_in_order": _single_item,
    "item_price_inr": _price,
    "mrp_inr": _price,
    "ev_empty_box": _flag("ev_empty_box", "claims an empty box"),
    "ev_tags_intact": _flag("ev_tags_intact", "says tags are intact"),
    "ev_worn_signs": _flag("ev_worn_signs", "text hints the item was used"),
    "ev_fit_small": _flag("ev_fit_small", "complains the item is too small"),
    "ev_fit_large": _flag("ev_fit_large", "complains the item is too large"),
    "ev_wrong_item": _flag("ev_wrong_item", "reports a wrong item"),
    "ev_damage": _flag("ev_damage", "reports damage"),
    "ev_refund_demand": _flag("ev_refund_demand", "demands a refund without a defect"),
    "ev_dissatisfied": _flag("ev_dissatisfied", "gives no concrete defect"),
    "txt_p_abusive": lambda row: "reason wording resembles abusive returns" if row["txt_p_abusive"] >= 0.2 else None,
    "txt_p_genuine_fit_issue": _text_genuine,
    "txt_p_genuine_quality_issue": _text_genuine,
    "txt_p_genuine_wrong_item": _text_genuine,
}


def _top_phrases(row: pd.Series, contribution: pd.Series, direction: int, limit: int) -> list[str]:
    """Phrases for the features that push the most in the requested direction (+1 abusive, -1 genuine)."""
    phrases: list[str] = []
    for name in (contribution * direction).sort_values(ascending=False).index:
        if contribution[name] * direction <= 0.05 or len(phrases) >= limit:
            break
        if direction > 0 and name.startswith("txt_p_genuine"):
            continue
        if direction < 0 and name == "txt_p_abusive":
            continue
        make = PHRASES.get(name)
        phrase = make(row) if make else None
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases


def justify(row: pd.Series, contribution: pd.Series, decision: str, p_abuse: float, limit: int = 3) -> str:
    """One or two sentences naming the decision, the abuse probability and the evidence behind it."""
    if decision == "auto_approve":
        reasons = _top_phrases(row, contribution, -1, limit) or ["no strong abuse signal in history, timing or text"]
        return f"Approve (P(abusive)={p_abuse:.0%}): " + "; ".join(reasons) + "."
    reasons = _top_phrases(row, contribution, +1, limit) or ["combined risk score is high"]
    verb = "Escalate" if decision == "escalate" else "Inspect"
    note = "" if decision == "escalate" else " Cheap check, not a rejection."
    return f"{verb} (P(abusive)={p_abuse:.0%}): " + "; ".join(reasons) + "." + note


def justify_all(
    rows: pd.DataFrame,
    contributions: pd.DataFrame,
    decisions: np.ndarray,
    p_abuse: np.ndarray,
    limit: int = 3,
) -> list[str]:
    """Justification for every row, aligned with ``rows``."""
    return [
        justify(rows.iloc[i], contributions.iloc[i], str(decisions[i]), float(p_abuse[i]), limit)
        for i in range(len(rows))
    ]
