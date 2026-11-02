"""Phase 3: engineered structured features and a preprocessing pipeline that is fitted on training rows only.

Two layers:
* ``add_engineered_features`` - stateless features. Row-level ones use only the row's own columns; history ones use only
  EARLIER rows of the same customer (the same rule as ``features.add_history_features``).
* ``FeaturePreprocessor`` - learned transformations (medians, scaling, one-hot vocabulary, frequency tables). They are
  fitted on the rows passed to ``fit`` and only applied by ``transform_*``, so later rows never influence them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# --- feature groups -----------------------------------------------------------------------------------------------

ORIGINAL_NUMERIC = (
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
CATALOGUE_NUMERIC = ("mrp_inr", "size_pos_frac", "price_mismatch", "item_share_of_order")
TIMING_NUMERIC = ("days_since_order", "return_window_remaining")
PRICE_NUMERIC = (
    "log_item_price",
    "log_order_value",
    "discount_amount_inr",
    "price_band",
    "order_value_band",
    "discount_band",
)
SIZE_NUMERIC = ("size_chart_deviation",)
CUSTOMER_NUMERIC = ("prior_return_count_est", "log1p_prior_orders")
HISTORY_NUMERIC = (
    "hist_n_prev",
    "hist_days_since_prev",
    "hist_returns_30d",
    "hist_returns_60d",
    "hist_returns_90d",
    "hist_same_product",
    "hist_same_subcat_size",
    "hist_same_category",
    "hist_same_order",
    "hist_mean_item_price",
    "hist_price_ratio",
    "hist_rate_gap",
    "refund_value_30d",
    "refund_value_60d",
    "refund_value_90d",
)
BINARY_FEATURES = (
    "cod_flag",
    "is_late_return",
    "is_high_discount",
    "is_weekend",
    "prior_return_rate_missing",
    "customer_history_available",
    "catalogue_size_match",
    "size_chart_missing",
    "invalid_date_order",
    "price_exceeds_mrp",
    "missing_item_price",
)
CATEGORICAL_FEATURES = (
    "category",
    "sub_category",
    "payment_method",
    "fulfilment_centre",
    "size_kind",
    "brand",
    "gender",
    "fabric",
    "fit_type",
    "request_dow",
)
FREQUENCY_SOURCES = ("product_id", "sub_category", "brand")
LEARNED_NUMERIC = ("product_freq", "sub_category_freq", "brand_freq", "price_vs_subcategory_median")
NUMERIC_FEATURES = (
    *ORIGINAL_NUMERIC,
    *CATALOGUE_NUMERIC,
    *TIMING_NUMERIC,
    *PRICE_NUMERIC,
    *SIZE_NUMERIC,
    *CUSTOMER_NUMERIC,
    *HISTORY_NUMERIC,
)
STATELESS_FEATURES = (*NUMERIC_FEATURES, *BINARY_FEATURES, *CATEGORICAL_FEATURES)
_SIZE_MEASURES = ("chest_cm", "waist_cm", "foot_length_cm", "length_cm")

# Columns that are deliberately NOT model inputs in Phase 3, with the reason.
EXCLUDED_COLUMNS: dict[str, str] = {
    "return_id": "identifier - carries no behaviour and would let a model memorise rows",
    "order_id": "identifier",
    "customer_id": "identifier (customer history is expressed through point-in-time features instead)",
    "product_id": "identifier (replaced by the training-fitted product_freq)",
    "order_date": "raw date - converted to days_since_order and request_dow",
    "delivery_date": "raw date - converted to timing features",
    "return_request_date": "raw date and the split clock; month is not used (the abuse rate drifts by month)",
    "label": "TARGET - never an input",
    "abusive": "TARGET-derived",
    "customer_reason": "free text - handled in Phase 4 (out-of-fold text probabilities)",
    "hist_max_text_sim": "text-derived - Phase 4",
    "hist_n_similar_text": "text-derived - Phase 4",
    "hist_same_text": "text-derived - Phase 4",
    "lab_n_prev": "earlier OUTCOMES (label history) - needs the review-delay assumption; decided later",
    "lab_n_abusive": "earlier outcomes - see above",
    "lab_abuse_share": "earlier outcomes - see above",
    "lab_n_fit": "earlier outcomes - see above",
    "lab_n_quality": "earlier outcomes - see above",
    "lab_n_wrong": "earlier outcomes - see above",
    "lab_days_since_abusive": "earlier outcomes - see above",
    "non_english": "fairness group - must never be a model input (proxy risk)",
    "is_new_customer": "fairness group flag - deliberately NOT created; raw tenure is kept instead",
    "is_low_value_order": "fairness group flag - deliberately NOT created; raw order value is kept instead",
    "is_late_return_outside_window": "constant: no request is later than the 30-day window (Phase 1)",
    "title": "catalogue free text, not needed",
    "colour": "catalogue attribute with no link to the outcome in Phase 1",
    "available_sizes": "raw string - converted to size_pos_frac and catalogue_size_match",
    "size_chart_id": "identifier",
    "is_train": "bookkeeping",
    "day": "bookkeeping index",
}


# --- stateless features -------------------------------------------------------------------------------------------


def _band(values: pd.Series, edges: list[float]) -> pd.Series:
    """Ordinal band 0..len(edges) from fixed, hand-chosen edges (not learned from data)."""
    return pd.Series(
        np.searchsorted(np.asarray(edges, dtype=float), values.to_numpy(dtype=float), side="right"), index=values.index
    )


def _refund_velocity(frame: pd.DataFrame, windows: list[int]) -> dict[str, np.ndarray]:
    """Sum of item prices of the customer's EARLIER requests inside each look-back window."""
    out = {f"refund_value_{w}d": np.zeros(len(frame)) for w in windows}
    days = frame["day"].to_numpy()
    prices = frame["item_price_inr"].fillna(0).to_numpy(dtype=float)
    for positions in frame.groupby("customer_id", sort=False).indices.values():
        rows = np.asarray(positions)
        d, cumulative = days[rows], np.r_[0.0, np.cumsum(prices[rows])]
        for w in windows:
            start = np.searchsorted(d, d - w, side="left")
            out[f"refund_value_{w}d"][rows] = cumulative[np.arange(len(rows))] - cumulative[start]
    return out


def _size_chart_deviation(frame: pd.DataFrame, size_charts: pd.DataFrame) -> pd.Series:
    """Relative gap between the ordered size's measurement and the chart's median size (first available measure)."""
    chart = size_charts.copy()
    for measure in _SIZE_MEASURES:
        chart[f"{measure}_med"] = chart.groupby("size_chart_id")[measure].transform("median")
    joined = frame[["size_chart_id", "size_ordered"]].merge(
        chart, left_on=["size_chart_id", "size_ordered"], right_on=["size_chart_id", "size"], how="left"
    )
    deviation = pd.Series(np.nan, index=range(len(joined)))
    for measure in reversed(_SIZE_MEASURES):  # earlier measures overwrite later ones
        gap = joined[measure] / joined[f"{measure}_med"] - 1
        deviation = gap.where(gap.notna(), deviation)
    return pd.Series(deviation.to_numpy(), index=frame.index)


def add_engineered_features(
    frame: pd.DataFrame,
    size_charts: pd.DataFrame,
    cfg: dict[str, Any],
    customers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add all stateless Phase 3 features. ``frame`` must come from ``features.build_feature_table``."""
    settings = cfg["preprocess"]
    out = frame.copy()
    window, late_after = int(settings["return_window_days"]), int(settings["late_after_days"])

    # timing
    out["days_since_order"] = (out["return_request_date"] - out["order_date"]).dt.days
    out["return_window_remaining"] = window - out["days_since_delivery"]
    out["is_late_return"] = (out["days_since_delivery"] > late_after).astype(int)
    out["request_dow"] = out["return_request_date"].dt.day_of_week.astype(str)
    out["is_weekend"] = (out["return_request_date"].dt.day_of_week >= 5).astype(int)

    # price
    out["log_item_price"] = np.log1p(out["item_price_inr"].clip(lower=0))
    out["log_order_value"] = np.log1p(out["order_value_inr"].clip(lower=0))
    out["discount_amount_inr"] = out["mrp_inr"] - out["item_price_inr"]
    out["price_band"] = _band(out["item_price_inr"], settings["price_edges_inr"])
    out["order_value_band"] = _band(out["order_value_inr"], settings["price_edges_inr"])
    out["discount_band"] = _band(out["discount_pct"], settings["discount_edges_pct"])
    out["is_high_discount"] = (out["discount_pct"] > settings["discount_edges_pct"][-1]).astype(int)
    out["price_exceeds_mrp"] = (out["item_price_inr"] > out["mrp_inr"] * 1.01).astype(int)

    # size and catalogue
    sizes = out["available_sizes"].fillna("").str.split("|")
    out["catalogue_size_match"] = [int(o in s) for s, o in zip(sizes, out["size_ordered"], strict=True)]
    out["size_chart_missing"] = out["size_chart_id"].isna().astype(int)
    out["size_chart_deviation"] = _size_chart_deviation(out, size_charts)

    # customer
    out["prior_return_rate_missing"] = out["prior_return_rate"].isna().astype(int)
    out["customer_history_available"] = (out["prior_orders_count"] > 0).astype(int)
    out["prior_return_count_est"] = out["prior_return_rate"].fillna(0) * out["prior_orders_count"]
    out["log1p_prior_orders"] = np.log1p(out["prior_orders_count"].clip(lower=0))
    for name, values in _refund_velocity(out, list(settings["velocity_windows_days"])).items():
        out[name] = values

    # customer table (optional): consistency check of the supplied tenure, and city_tier kept only as a CANDIDATE
    if customers is not None:
        info = customers[["customer_id", "signup_date", "city_tier"]]
        out = out.merge(info, on="customer_id", how="left", validate="many_to_one")
        recomputed = (out["return_request_date"] - out["signup_date"]).dt.days
        out["tenure_gap_days"] = recomputed - out["customer_tenure_days"]

    # data quality
    dates_ok = (out["order_date"] <= out["delivery_date"]) & (out["delivery_date"] <= out["return_request_date"])
    out["invalid_date_order"] = (~dates_ok.fillna(False)).astype(int)
    out["missing_item_price"] = out["item_price_inr"].isna().astype(int)
    return out


# --- learned preprocessing ----------------------------------------------------------------------------------------


class FeaturePreprocessor:
    """Learns imputation, scaling, one-hot vocabularies and frequency tables from TRAINING rows only."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._min_count = int(cfg["preprocess"]["rare_category_min_count"])
        self._frequency: dict[str, pd.Series] = {}
        self._subcategory_median_price: pd.Series = pd.Series(dtype=float)
        self._categories: dict[str, set[str]] = {}
        self._linear: ColumnTransformer | None = None
        self._numeric_columns = [*NUMERIC_FEATURES, *LEARNED_NUMERIC]
        self.n_training_rows = 0
        self.dropped_columns_: dict[str, str] = {}

    # learned columns -----------------------------------------------------
    def _add_learned(self, rows: pd.DataFrame) -> pd.DataFrame:
        out = rows.copy()
        for source in FREQUENCY_SOURCES:
            suffix = "product" if source == "product_id" else source
            out[f"{suffix}_freq"] = out[source].astype(object).map(self._frequency[source]).astype(float).fillna(0.0)
        median = out["sub_category"].astype(object).map(self._subcategory_median_price).astype(float)
        out["price_vs_subcategory_median"] = out["item_price_inr"] / median
        return out

    def unseen_category_count(self, rows: pd.DataFrame) -> pd.Series:
        """How many categorical values of each row never occurred in the training rows."""
        unseen = [~rows[c].astype(str).isin(self._categories[c]) for c in CATEGORICAL_FEATURES]
        return pd.concat(unseen, axis=1).sum(axis=1).astype(int)

    # API -----------------------------------------------------------------
    def fit(self, rows: pd.DataFrame) -> FeaturePreprocessor:
        """Learn every parameter from ``rows`` (training rows only)."""
        if rows.empty:
            raise ValueError("cannot fit the preprocessor on zero rows")
        missing = [c for c in STATELESS_FEATURES if c not in rows.columns]
        if missing:
            raise ValueError(f"missing engineered columns, call add_engineered_features first: {missing[:5]}")
        self.n_training_rows = len(rows)
        for source in FREQUENCY_SOURCES:
            self._frequency[source] = rows[source].value_counts(normalize=True)
        self._subcategory_median_price = rows.groupby("sub_category")["item_price_inr"].median()
        self._categories = {c: set(rows[c].astype(str).unique()) for c in CATEGORICAL_FEATURES}
        prepared = self._add_learned(rows)
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                ("scale", RobustScaler()),
            ]
        )
        categorical = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=self._min_count, sparse_output=False
        )
        self._linear = ColumnTransformer(
            [
                ("numeric", numeric, self._numeric_columns),
                ("binary", "passthrough", list(BINARY_FEATURES)),
                ("categorical", categorical, list(CATEGORICAL_FEATURES)),
            ],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
        self._linear.fit(prepared.assign(**{c: prepared[c].astype(str) for c in CATEGORICAL_FEATURES}))
        self.dropped_columns_ = self._find_redundant_columns(self._raw_linear(rows))
        return self

    @staticmethod
    def _find_redundant_columns(matrix: pd.DataFrame) -> dict[str, str]:
        """Constant columns and exact duplicates of an earlier column, judged on the TRAINING matrix."""
        dropped: dict[str, str] = {}
        seen: dict[int, str] = {}
        for name in matrix.columns:
            values = matrix[name]
            if values.nunique(dropna=False) <= 1:
                dropped[name] = "constant in the training rows"
                continue
            key = hash(values.to_numpy().tobytes())
            if key in seen and np.array_equal(matrix[seen[key]].to_numpy(), values.to_numpy(), equal_nan=True):
                dropped[name] = f"exact duplicate of {seen[key]}"
            else:
                seen[key] = name
        return dropped

    def _raw_linear(self, rows: pd.DataFrame) -> pd.DataFrame:
        prepared = self._add_learned(rows)
        return self._linear.transform(prepared.assign(**{c: prepared[c].astype(str) for c in CATEGORICAL_FEATURES}))

    def transform_linear(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Fully numeric, imputed and scaled matrix with one-hot categories (for linear models). No NaNs."""
        if self._linear is None:
            raise RuntimeError("FeaturePreprocessor must be fitted first")
        matrix = self._raw_linear(rows).drop(columns=list(self.dropped_columns_))
        return pd.concat([matrix, self.unseen_category_count(rows).rename("n_unseen_categories")], axis=1)

    def transform_tree(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Raw numeric values (NaN kept, trees handle it) plus categoricals typed with the TRAINING categories."""
        if not self._categories:
            raise RuntimeError("FeaturePreprocessor must be fitted first")
        prepared = self._add_learned(rows)
        matrix = prepared[[*self._numeric_columns, *BINARY_FEATURES]].drop(
            columns=list(self.dropped_columns_), errors="ignore"
        )
        for column in CATEGORICAL_FEATURES:
            known = sorted(self._categories[column])
            values = prepared[column].astype(str)
            matrix[column] = pd.Categorical(values.where(values.isin(known)), categories=known)
        matrix["n_unseen_categories"] = self.unseen_category_count(rows).to_numpy()
        return matrix

    def learned_parameters(self) -> dict[str, Any]:
        """Human-readable summary of what was learned (no pickling needed)."""
        if self._linear is None:
            raise RuntimeError("FeaturePreprocessor must be fitted first")
        imputer = self._linear.named_transformers_["numeric"].named_steps["impute"]
        scaler = self._linear.named_transformers_["numeric"].named_steps["scale"]
        return {
            "training_rows": self.n_training_rows,
            "numeric_medians": dict(zip(self._numeric_columns, np.round(imputer.statistics_, 4), strict=True)),
            "robust_scaler_center": dict(
                zip(self._numeric_columns, np.round(scaler.center_[: len(self._numeric_columns)], 4), strict=True)
            ),
            "categories": {c: sorted(v) for c, v in self._categories.items()},
            "frequency_table_sizes": {k: len(v) for k, v in self._frequency.items()},
            "dropped_columns": dict(self.dropped_columns_),
        }


# --- reporting ----------------------------------------------------------------------------------------------------


def feature_quality_report(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-column data type, missing share, number of distinct values and a constant/duplicate flag."""
    numeric = matrix.select_dtypes("number")
    duplicates = numeric.T.duplicated(keep="first")
    report = pd.DataFrame(
        {
            "dtype": matrix.dtypes.astype(str),
            "missing_%": (matrix.isna().mean() * 100).round(3),
            "distinct": matrix.nunique(dropna=True),
        }
    )
    report["constant"] = report["distinct"] <= 1
    report["duplicate_of_earlier_column"] = duplicates.reindex(report.index, fill_value=False)
    report["has_inf"] = numeric.replace([np.inf, -np.inf], np.nan).isna().sum().reindex(
        report.index
    ) - numeric.isna().sum().reindex(report.index)
    report["has_inf"] = report["has_inf"].fillna(0).astype(int) > 0
    return report


def data_quality_flags(rows: pd.DataFrame) -> pd.Series:
    """Counts of rows that violate a data-quality rule (monitoring guards; all zero in the original data)."""
    return pd.Series(
        {
            "invalid_date_order": int(rows["invalid_date_order"].sum()),
            "missing_item_price": int(rows["missing_item_price"].sum()),
            "price_exceeds_mrp": int(rows["price_exceeds_mrp"].sum()),
            "size_not_in_catalogue": int((rows["catalogue_size_match"] == 0).sum()),
            "outside_return_window": int((rows["return_window_remaining"] < 0).sum()),
            "tenure_mismatch_over_1_day": int((rows["tenure_gap_days"].abs() > 1).sum())
            if "tenure_gap_days" in rows
            else -1,
        }
    )


def validate_joins(
    requests: pd.DataFrame, products: pd.DataFrame, customers: pd.DataFrame, size_charts: pd.DataFrame
) -> pd.Series:
    """Join coverage and key uniqueness, checked BEFORE merging (a failed check must stop the pipeline)."""
    charted = products["size_chart_id"].dropna()
    return pd.Series(
        {
            "product_id_duplicates": int(products["product_id"].duplicated().sum()),
            "customer_id_duplicates": int(customers["customer_id"].duplicated().sum()),
            "requests_without_product": int((~requests["product_id"].isin(products["product_id"])).sum()),
            "requests_without_customer": int((~requests["customer_id"].isin(customers["customer_id"])).sum()),
            "products_with_unknown_chart": int((~charted.isin(size_charts["size_chart_id"])).sum()),
            "products_without_chart": int(products["size_chart_id"].isna().sum()),
        }
    )


# --- documentation of every feature -------------------------------------------------------------------------------

_ROW = "row's own columns only"
_PAST = "uses only EARLIER requests of the same customer"
_STATIC = "static catalogue fact"
_FIXED = "fixed hand-chosen edges, nothing learned"
_TRAIN = "learned from training rows only"

_CATALOGUE: tuple[tuple[str, str, str, str, str], ...] = (
    ("gender", "product", "products.gender", "one-hot product gender", _STATIC),
    # name, group, source columns, rule, leakage safety
    ("delivery_days", "original", "delivery_days", "as given", _ROW),
    ("days_since_delivery", "original", "days_since_delivery", "as given", _ROW),
    ("order_value_inr", "original", "order_value_inr", "as given", _ROW),
    ("item_price_inr", "original", "item_price_inr", "as given", _ROW),
    ("discount_pct", "original", "discount_pct", "as given", _ROW),
    ("items_in_order", "original", "items_in_order", "as given", _ROW),
    ("customer_tenure_days", "original", "customer_tenure_days", "as given (also a fairness group)", _ROW),
    ("prior_orders_count", "original", "prior_orders_count", "as given", "given by the retailer at request time"),
    (
        "prior_return_rate",
        "original",
        "prior_return_rate",
        "as given; missing = no earlier orders",
        "given at request time",
    ),
    ("mrp_inr", "catalogue", "products.price_inr", "catalogue price", _STATIC),
    (
        "size_pos_frac",
        "catalogue",
        "size_ordered, available_sizes",
        "position of the ordered size in the size run",
        _STATIC,
    ),
    ("price_mismatch", "catalogue", "item_price, mrp, discount", "|price - mrp*(1-discount)| / mrp", _ROW),
    ("item_share_of_order", "catalogue", "item_price, order_value", "item price / order value", _ROW),
    ("days_since_order", "timing", "order_date, return_request_date", "days from order to return request", _ROW),
    ("return_window_remaining", "timing", "days_since_delivery", "30 - days since delivery", _ROW),
    ("log_item_price", "price", "item_price_inr", "log1p", _ROW),
    ("log_order_value", "price", "order_value_inr", "log1p", _ROW),
    ("discount_amount_inr", "price", "mrp, item_price", "mrp - item price", _ROW),
    ("price_band", "price", "item_price_inr", "ordinal band 0-4 (<500, <1000, <2000, <3000, more)", _FIXED),
    ("order_value_band", "price", "order_value_inr", "ordinal band 0-4, same edges", _FIXED),
    ("discount_band", "price", "discount_pct", "ordinal band 0-3 (<10, <30, <50, more)", _FIXED),
    (
        "size_chart_deviation",
        "size",
        "size chart, size_ordered",
        "ordered size's measurement / chart median - 1",
        _STATIC,
    ),
    (
        "prior_return_count_est",
        "customer",
        "prior_return_rate, prior_orders_count",
        "rate x orders (0 when missing)",
        _ROW,
    ),
    ("log1p_prior_orders", "customer", "prior_orders_count", "log1p", _ROW),
    ("hist_n_prev", "history", "customer_id, date order", "earlier requests by the customer", _PAST),
    ("hist_days_since_prev", "history", "customer_id, day", "days since the previous request", _PAST),
    ("hist_returns_30d", "history", "customer_id, day", "earlier requests in the last 30 days", _PAST),
    ("hist_returns_60d", "history", "customer_id, day", "earlier requests in the last 60 days", _PAST),
    ("hist_returns_90d", "history", "customer_id, day", "earlier requests in the last 90 days", _PAST),
    ("hist_same_product", "history", "customer_id, product_id", "earlier requests for the same product", _PAST),
    (
        "hist_same_subcat_size",
        "history",
        "customer, sub_category, size",
        "earlier requests, same sub-category and size",
        _PAST,
    ),
    ("hist_same_category", "history", "customer, category", "earlier requests in the same category", _PAST),
    ("hist_same_order", "history", "order_id", "earlier requests from the same order", _PAST),
    ("hist_mean_item_price", "history", "customer, item_price", "mean item price of earlier requests", _PAST),
    ("hist_price_ratio", "history", "item_price, hist_mean_item_price", "this item's price / earlier mean", _PAST),
    ("hist_rate_gap", "history", "hist_n_prev, prior_orders_count", "earlier requests / prior orders", _PAST),
    (
        "refund_value_30d",
        "history",
        "customer, item_price",
        "refund-value velocity: sum of earlier item prices, 30 d",
        _PAST,
    ),
    ("refund_value_60d", "history", "customer, item_price", "same, 60 d", _PAST),
    ("refund_value_90d", "history", "customer, item_price", "same, 90 d", _PAST),
    ("cod_flag", "binary", "payment_method", "1 if cash on delivery", _ROW),
    ("is_late_return", "binary", "days_since_delivery", "1 if more than 14 days after delivery", _ROW),
    ("is_high_discount", "binary", "discount_pct", "1 if discount above 50%", _ROW),
    ("is_weekend", "binary", "return_request_date", "1 on Saturday or Sunday", _ROW),
    (
        "prior_return_rate_missing",
        "missingness",
        "prior_return_rate",
        "1 if the rate is missing (first-time customer)",
        _ROW,
    ),
    ("customer_history_available", "missingness", "prior_orders_count", "1 if the customer has earlier orders", _ROW),
    ("catalogue_size_match", "data quality", "size_ordered, available_sizes", "1 if the ordered size is sold", _STATIC),
    ("size_chart_missing", "data quality", "size_chart_id", "1 if the product has no chart (free size)", _STATIC),
    ("invalid_date_order", "data quality", "three dates", "1 if order > delivery or delivery > request", _ROW),
    (
        "price_exceeds_mrp",
        "data quality",
        "item_price, mrp",
        "1 if price is more than 1% above the catalogue price",
        _ROW,
    ),
    ("missing_item_price", "data quality", "item_price_inr", "1 if the price is missing", _ROW),
    ("category", "categorical", "category", "one-hot, rare values pooled", _TRAIN),
    ("sub_category", "categorical", "sub_category", "one-hot, rare values pooled", _TRAIN),
    ("payment_method", "categorical", "payment_method", "one-hot", _TRAIN),
    ("fulfilment_centre", "categorical", "fulfilment_centre", "one-hot", _TRAIN),
    ("size_kind", "categorical", "size_ordered", "free / shoe / waist / alpha", _ROW),
    ("brand", "categorical", "products.brand", "one-hot, rare values pooled", _TRAIN),
    ("fabric", "categorical", "products.fabric", "one-hot, rare values pooled", _TRAIN),
    ("fit_type", "categorical", "products.fit_type", "one-hot, rare values pooled", _TRAIN),
    ("request_dow", "categorical", "return_request_date", "weekday of the request", _ROW),
    ("product_freq", "learned", "product_id", "share of training rows for this product (unseen = 0)", _TRAIN),
    ("sub_category_freq", "learned", "sub_category", "share of training rows (unseen = 0)", _TRAIN),
    ("brand_freq", "learned", "brand", "share of training rows (unseen = 0)", _TRAIN),
    (
        "price_vs_subcategory_median",
        "learned",
        "item_price, sub_category",
        "price / training median of the sub-category",
        _TRAIN,
    ),
    (
        "n_unseen_categories",
        "data quality",
        "all categorical columns",
        "categorical values never seen in training",
        _TRAIN,
    ),
)


def feature_catalogue() -> pd.DataFrame:
    """Documented list of every engineered/original structured feature, its source, rule and leakage safety."""
    return pd.DataFrame(_CATALOGUE, columns=["feature", "group", "source", "rule", "leakage_safety"]).set_index(
        "feature"
    )


def excluded_columns() -> pd.DataFrame:
    """Columns that are deliberately not inputs, with the reason."""
    return pd.DataFrame({"reason": pd.Series(EXCLUDED_COLUMNS)}).rename_axis("column")
