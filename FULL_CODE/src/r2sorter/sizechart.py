"""Which products have the most fit-related returns? (the size-chart top 5)."""

from __future__ import annotations

from typing import Any

import pandas as pd

FIT = "genuine_fit_issue"


def fit_return_table(
    train: pd.DataFrame, test: pd.DataFrame, test_fit_probability: pd.Series, products: pd.DataFrame
) -> pd.DataFrame:
    """Per product: true fit returns in train, predicted fit returns in test, and a combined monthly rate."""
    train_months = (train["return_request_date"].max() - train["return_request_date"].min()).days / 30.4 + 1
    test_months = (test["return_request_date"].max() - test["return_request_date"].min()).days / 30.4 + 1
    table = pd.DataFrame(
        {
            "train_fit_returns": train[train["label"] == FIT].groupby("product_id").size(),
            "test_expected_fit_returns": test_fit_probability.groupby(test["product_id"]).sum(),
            "train_returns": train.groupby("product_id").size(),
            "test_returns": test.groupby("product_id").size(),
        }
    ).fillna(0.0)
    total_months = train_months + test_months
    table["combined_fit_returns"] = table["train_fit_returns"] + table["test_expected_fit_returns"]
    table["fit_returns_per_month"] = table["combined_fit_returns"] / total_months
    small = train[train["label"] == FIT].assign(
        small=lambda d: d["customer_reason"].str.contains(r"small|tight|chhota|chota|tang|kasa", case=False, regex=True)
    )
    table["train_small_share"] = small.groupby("product_id")["small"].mean()
    return table.join(products.set_index("product_id")[["title", "category", "sub_category", "size_chart_id"]])


def top_products(table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Top-k products by combined fit-return volume, ranked from 1."""
    eligible = table[(table["train_returns"] + table["test_returns"]) >= cfg["size_chart"]["min_returns"]]
    best = eligible.sort_values("combined_fit_returns", ascending=False).head(int(cfg["size_chart"]["top_k"]))
    return best.reset_index(names="product_id").assign(rank=lambda d: range(1, len(d) + 1))
