"""Loading and validating the official return-request files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import data_file

LOGGER = logging.getLogger(__name__)

LABELS = ("abusive", "genuine_quality_issue", "genuine_fit_issue", "genuine_wrong_item")
DECISIONS = ("auto_approve", "inspect", "escalate")

REQUEST_COLUMNS = (
    "return_id",
    "order_id",
    "customer_id",
    "product_id",
    "category",
    "sub_category",
    "size_ordered",
    "order_date",
    "delivery_date",
    "return_request_date",
    "delivery_days",
    "days_since_delivery",
    "order_value_inr",
    "item_price_inr",
    "discount_pct",
    "items_in_order",
    "payment_method",
    "fulfilment_centre",
    "customer_tenure_days",
    "prior_orders_count",
    "prior_return_rate",
    "customer_reason",
)
DATE_COLUMNS = ("order_date", "delivery_date", "return_request_date")


class DataError(ValueError):
    """Raised when an input file does not look like the official data."""


def read_requests(path: Path, *, with_label: bool) -> pd.DataFrame:
    """Read one return-request CSV, check its schema and parse dates."""
    if not path.exists():
        raise DataError(f"Data file not found: {path}. See the README for where to put the data.")
    frame = pd.read_csv(path)
    required = list(REQUEST_COLUMNS) + (["label"] if with_label else [])
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise DataError(f"{path.name} is missing columns: {missing}")
    if frame["return_id"].duplicated().any():
        raise DataError(f"{path.name} has duplicate return_id values")
    if with_label:
        unknown = sorted(set(frame["label"]) - set(LABELS))
        if unknown:
            raise DataError(f"{path.name} has unknown labels: {unknown}")
    for column in DATE_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="raise")
    frame["customer_reason"] = frame["customer_reason"].fillna("").astype(str)
    return frame


def load_all(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Load train, test, products, size charts and customers."""
    train = read_requests(data_file(cfg, "train_file"), with_label=True)
    test = read_requests(data_file(cfg, "test_file"), with_label=False)
    if set(train["return_id"]) & set(test["return_id"]):
        raise DataError("train and test share return_id values")
    LOGGER.info("loaded %d train and %d test requests", len(train), len(test))
    return {
        "train": train,
        "test": test,
        "products": pd.read_csv(data_file(cfg, "products_file")),
        "size_charts": pd.read_csv(data_file(cfg, "size_charts_file")),
        "customers": pd.read_csv(data_file(cfg, "customers_file"), parse_dates=["signup_date"]),
    }
