"""Point-in-time feature engineering.

Every history feature for a request only looks at requests that came EARLIER in the data
(sorted by request date, then id), never at the customer's future. Labels are never used here.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .text import evidence_features, normalise_reason

CATEGORICAL_COLUMNS = (
    "category",
    "sub_category",
    "payment_method",
    "fulfilment_centre",
    "size_kind",
    "brand",
    "fabric",
    "fit_type",
)
BASE_NUMERIC = (
    "delivery_days",
    "days_since_delivery",
    "order_value_inr",
    "item_price_inr",
    "discount_pct",
    "items_in_order",
    "customer_tenure_days",
    "prior_orders_count",
    "prior_return_rate",
)
_WAIST = re.compile(r"^\d{2}$")


def combine_requests(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Stack train and test in time order so history features see exactly what was known at request time."""
    stacked = pd.concat([train.assign(is_train=True), test.assign(is_train=False, label=np.nan)], ignore_index=True)
    stacked = stacked.sort_values(["return_request_date", "return_id"], kind="stable").reset_index(drop=True)
    stacked["day"] = (stacked["return_request_date"] - stacked["return_request_date"].min()).dt.days
    return stacked


def _size_kind(size: str) -> str:
    if size == "Free Size":
        return "free"
    if size.startswith("UK"):
        return "shoe"
    return "waist" if _WAIST.match(size) else "alpha"


def add_catalogue_features(frame: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Join product facts and describe which size the customer picked."""
    keep = [
        "product_id",
        "brand",
        "gender",
        "price_inr",
        "available_sizes",
        "fabric",
        "fit_type",
        "size_chart_id",
    ]
    merged = frame.merge(products[keep], on="product_id", how="left", validate="many_to_one")
    merged["size_kind"] = merged["size_ordered"].map(_size_kind)
    sizes = merged["available_sizes"].fillna("").str.split("|")
    position = [s.index(o) if o in s else -1 for s, o in zip(sizes, merged["size_ordered"], strict=True)]
    count = sizes.str.len().clip(lower=2)
    merged["size_pos_frac"] = np.where(np.array(position) >= 0, np.array(position) / (count - 1), np.nan)
    merged["mrp_inr"] = merged["price_inr"]
    expected_price = merged["mrp_inr"] * (1 - merged["discount_pct"] / 100)
    merged["price_mismatch"] = (merged["item_price_inr"] - expected_price).abs() / merged["mrp_inr"].clip(lower=1)
    merged["item_share_of_order"] = merged["item_price_inr"] / merged["order_value_inr"].clip(lower=1)
    merged["cod_flag"] = (merged["payment_method"] == "COD").astype(int)
    return merged


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def add_history_features(frame: pd.DataFrame, windows: list[int], lookback: int) -> pd.DataFrame:
    """Counts and repeat patterns from the customer's EARLIER requests."""
    out = frame.copy()
    by_customer = out.groupby("customer_id", sort=False)
    out["hist_n_prev"] = by_customer.cumcount()
    out["hist_days_since_prev"] = by_customer["day"].diff()
    for window in windows:
        out[f"hist_returns_{window}d"] = 0
    out["hist_max_text_sim"] = 0.0
    out["hist_n_similar_text"] = 0
    tokens = [frozenset(normalise_reason(r).split()) for r in out["customer_reason"]]
    days = out["day"].to_numpy()
    for positions in by_customer.indices.values():
        rows = np.asarray(positions)
        d = days[rows]
        for window in windows:
            start = np.searchsorted(d, d - window, side="left")
            out.loc[out.index[rows], f"hist_returns_{window}d"] = np.arange(len(rows)) - start
        if len(rows) > 1:
            _fill_similarity(out, rows, tokens, lookback)
    out["hist_same_product"] = out.groupby(["customer_id", "product_id"]).cumcount()
    out["hist_same_subcat_size"] = out.groupby(["customer_id", "sub_category", "size_ordered"]).cumcount()
    out["hist_same_category"] = out.groupby(["customer_id", "category"]).cumcount()
    out["hist_same_order"] = out.groupby("order_id").cumcount()
    out["hist_same_text"] = out.groupby(["customer_id", out["customer_reason"].map(normalise_reason)]).cumcount()
    running = out.groupby("customer_id", sort=False)["item_price_inr"].cumsum() - out["item_price_inr"]
    out["hist_mean_item_price"] = running / out["hist_n_prev"].replace(0, np.nan)
    out["hist_price_ratio"] = out["item_price_inr"] / out["hist_mean_item_price"]
    out["hist_rate_gap"] = out["hist_n_prev"] / out["prior_orders_count"].clip(lower=1)
    return out


def _fill_similarity(out: pd.DataFrame, rows: np.ndarray, tokens: list[frozenset[str]], lookback: int) -> None:
    best = np.zeros(len(rows))
    similar = np.zeros(len(rows), dtype=int)
    for i in range(1, len(rows)):
        sims = [_jaccard(tokens[rows[i]], tokens[rows[j]]) for j in range(max(0, i - lookback), i)]
        best[i] = max(sims)
        similar[i] = sum(s >= 0.5 for s in sims)
    labels = out.index[rows]
    out.loc[labels, "hist_max_text_sim"] = best
    out.loc[labels, "hist_n_similar_text"] = similar


LABEL_HISTORY_COLUMNS = (
    "lab_n_prev",
    "lab_n_abusive",
    "lab_abuse_share",
    "lab_n_fit",
    "lab_n_quality",
    "lab_n_wrong",
    "lab_days_since_abusive",
)
_HISTORY_LABELS = ("abusive", "genuine_fit_issue", "genuine_quality_issue", "genuine_wrong_item")


def _visible_cutoffs(frame: pd.DataFrame, unlabelled_from: pd.Timestamp, max_lag_days: int, seed: int):
    """Per row: the last day whose outcomes the retailer could already know, and which rows carry outcomes."""
    first_date = frame["return_request_date"].min()
    first_day = int(frame["day"].min())
    cutoff_day = (unlabelled_from - first_date).days + first_day
    rng = np.random.default_rng(seed)
    day = frame["day"].to_numpy()
    lags = rng.integers(1, max_lag_days + 1, len(frame))
    cutoff = np.where(day >= cutoff_day, cutoff_day - 1, day - lags)
    known = (frame["is_train"] & (day < cutoff_day)).to_numpy()
    return cutoff, known


def add_label_history(frame: pd.DataFrame, unlabelled_from: pd.Timestamp, max_lag_days: int, seed: int) -> pd.DataFrame:
    """Add what the retailer already KNEW about the customer's earlier outcomes.

    Outcomes (the label) only become known after a review delay. For rows before ``unlabelled_from`` we draw a
    random delay between 1 and ``max_lag_days`` so the model sees the same staleness it will meet on new
    requests; rows on/after ``unlabelled_from`` can only see outcomes before that date. Outcomes from the
    request itself or later are never visible.
    """
    out = frame.copy()
    cutoff, known = _visible_cutoffs(out, unlabelled_from, max_lag_days, seed)
    day = out["day"].to_numpy()
    is_label = {name: known & (out["label"] == name).to_numpy() for name in _HISTORY_LABELS}
    counts = {name: np.zeros(len(out)) for name in (*_HISTORY_LABELS, "all")}
    since_abusive = np.full(len(out), np.nan)
    for positions in out.groupby("customer_id", sort=False).indices.values():
        rows = np.asarray(positions)
        d = day[rows]
        visible = np.searchsorted(d, np.minimum(cutoff[rows], d - 1), side="right")
        counts["all"][rows] = np.r_[0, np.cumsum(known[rows])][visible]
        for name in _HISTORY_LABELS:
            counts[name][rows] = np.r_[0, np.cumsum(is_label[name][rows])][visible]
        abusive_days = d[is_label["abusive"][rows]]
        if len(abusive_days):
            last = np.searchsorted(abusive_days, np.minimum(cutoff[rows], d - 1), side="right") - 1
            since_abusive[rows] = np.where(last >= 0, d - abusive_days[np.maximum(last, 0)], np.nan)
    out["lab_n_prev"] = counts["all"]
    out["lab_n_abusive"] = counts["abusive"]
    out["lab_abuse_share"] = np.where(counts["all"] > 0, counts["abusive"] / np.maximum(counts["all"], 1), np.nan)
    out["lab_n_fit"] = counts["genuine_fit_issue"]
    out["lab_n_quality"] = counts["genuine_quality_issue"]
    out["lab_n_wrong"] = counts["genuine_wrong_item"]
    out["lab_days_since_abusive"] = since_abusive
    return out


def build_feature_table(
    train: pd.DataFrame, test: pd.DataFrame, products: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Full table (train + test rows) with all non-text-model features, in time order."""
    frame = combine_requests(train, test)
    frame = add_catalogue_features(frame, products)
    frame = add_history_features(frame, cfg["history_windows_days"], cfg["similarity_lookback"])
    frame = pd.concat([frame, evidence_features(frame["customer_reason"])], axis=1)
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype("category")
    return frame


def tabular_columns(frame: pd.DataFrame) -> list[str]:
    """Names of every model input column except the text-model probabilities."""
    derived = [
        "size_pos_frac",
        "mrp_inr",
        "price_mismatch",
        "item_share_of_order",
        "cod_flag",
        "hist_n_prev",
        "hist_days_since_prev",
        "hist_max_text_sim",
        "hist_n_similar_text",
        "hist_same_product",
        "hist_same_subcat_size",
        "hist_same_category",
        "hist_same_order",
        "hist_same_text",
        "hist_mean_item_price",
        "hist_price_ratio",
        "hist_rate_gap",
    ]
    windows = [c for c in frame.columns if re.fullmatch(r"hist_returns_\d+d", c)]
    labelled = [c for c in LABEL_HISTORY_COLUMNS if c in frame.columns]
    evidence = [c for c in frame.columns if c.startswith(("ev_", "txt_len", "txt_n_", "txt_upper", "txt_has"))]
    return [*BASE_NUMERIC, *derived, *windows, *labelled, *evidence, *CATEGORICAL_COLUMNS]
