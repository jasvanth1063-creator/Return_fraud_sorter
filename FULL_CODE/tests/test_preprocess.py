import numpy as np
import pandas as pd
import pytest

from r2sorter.preprocess import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    LEARNED_NUMERIC,
    NUMERIC_FEATURES,
    STATELESS_FEATURES,
    FeaturePreprocessor,
    add_engineered_features,
    data_quality_flags,
    feature_catalogue,
    feature_quality_report,
    validate_joins,
)


def _raw(prices=(100, 200, 300), days=(0, 10, 40)) -> pd.DataFrame:
    n = len(prices)
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(list(days), unit="D")
    return pd.DataFrame(
        {
            "customer_id": ["A"] * n,
            "day": list(days),
            "return_request_date": dates,
            "order_date": dates - pd.Timedelta(days=6),
            "delivery_date": dates - pd.Timedelta(days=2),
            "days_since_delivery": [2] * n,
            "item_price_inr": list(prices),
            "order_value_inr": [p * 2 for p in prices],
            "discount_pct": [0, 20, 60][:n],
            "mrp_inr": [100, 250, 750][:n],
            "available_sizes": ["S|M|L"] * n,
            "size_ordered": ["S", "M", "L"][:n],
            "size_chart_id": ["SC1"] * n,
            "prior_return_rate": [np.nan, 0.5, 0.5][:n],
            "prior_orders_count": [0, 4, 4][:n],
        }
    )


CHARTS = pd.DataFrame(
    {
        "size_chart_id": ["SC1"] * 3,
        "size": ["S", "M", "L"],
        "chest_cm": [90.0, 100.0, 110.0],
        "waist_cm": [np.nan] * 3,
        "foot_length_cm": [np.nan] * 3,
        "length_cm": [60.0, 62.0, 64.0],
    }
)


def test_refund_velocity_counts_only_earlier_rows(cfg):
    out = add_engineered_features(_raw(), CHARTS, cfg)
    assert out["refund_value_30d"].tolist() == [0, 100, 200]  # 3rd row (day 40) sees only day 10
    assert out["refund_value_60d"].tolist() == [0, 100, 300]


def test_future_rows_do_not_change_earlier_features(cfg):
    full = add_engineered_features(_raw(), CHARTS, cfg)
    short = add_engineered_features(_raw(prices=(100, 200), days=(0, 10)), CHARTS, cfg)
    cols = ["refund_value_30d", "refund_value_60d", "refund_value_90d"]
    pd.testing.assert_frame_equal(full.loc[:1, cols], short[cols], check_dtype=False)


def test_size_chart_deviation_uses_median_size(cfg):
    out = add_engineered_features(_raw(), CHARTS, cfg)
    assert out["size_chart_deviation"].tolist() == pytest.approx([-0.1, 0.0, 0.1])


def test_missingness_bands_and_flags(cfg):
    out = add_engineered_features(_raw(), CHARTS, cfg)
    assert out["prior_return_rate_missing"].tolist() == [1, 0, 0]
    assert out["customer_history_available"].tolist() == [0, 1, 1]
    assert out["prior_return_count_est"].tolist() == [0.0, 2.0, 2.0]  # missing rate is not read as suspicious
    assert out["price_band"].tolist() == [0, 0, 0]
    assert out["discount_band"].tolist() == [0, 1, 3]
    assert out["is_high_discount"].tolist() == [0, 0, 1]
    assert out["return_window_remaining"].tolist() == [28, 28, 28]
    assert out["days_since_order"].tolist() == [6, 6, 6]


def test_data_quality_flags_catch_bad_rows(cfg):
    bad = _raw()
    bad.loc[0, "delivery_date"] = bad.loc[0, "return_request_date"] + pd.Timedelta(days=3)
    bad.loc[1, "item_price_inr"] = np.nan
    bad.loc[2, "size_ordered"] = "XXL"
    flags = data_quality_flags(add_engineered_features(bad, CHARTS, cfg))
    assert flags["invalid_date_order"] == 1
    assert flags["missing_item_price"] == 1
    assert flags["size_not_in_catalogue"] == 1


def _training_rows(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC_FEATURES})
    for c in BINARY_FEATURES:
        rows[c] = rng.integers(0, 2, n)
    rows["cod_flag"] = rows["catalogue_size_match"] = 1  # constant columns
    for c in CATEGORICAL_FEATURES:
        rows[c] = rng.choice(["a", "b", "c"], n)
    rows["product_id"] = rng.choice(["P1", "P2", "P3"], n)
    rows.loc[::10, "prior_return_rate"] = np.nan
    rows["item_price_inr"] = rng.uniform(100, 900, n)
    return rows


def test_preprocessor_learns_only_from_the_fit_rows(cfg):
    rows = _training_rows()
    pre = FeaturePreprocessor(cfg).fit(rows)
    before = pre.learned_parameters()
    later = rows.copy()
    later["item_price_inr"] = 1e6  # extreme later data must not move anything that was learned
    later["sub_category"] = "never_seen"
    matrix = pre.transform_linear(later)
    assert pre.learned_parameters() == before
    assert matrix.isna().sum().sum() == 0
    assert (matrix["n_unseen_categories"] >= 1).all()


def test_constant_columns_are_dropped_and_reported(cfg):
    pre = FeaturePreprocessor(cfg).fit(_training_rows())
    assert "cod_flag" in pre.dropped_columns_
    assert "cod_flag" not in pre.transform_linear(_training_rows()).columns


def test_linear_and_tree_matrices_keep_training_categories(cfg):
    rows = _training_rows()
    pre = FeaturePreprocessor(cfg).fit(rows)
    tree = pre.transform_tree(rows.assign(brand="zzz"))
    assert set(tree["brand"].cat.categories) == {"a", "b", "c"}
    assert tree["brand"].isna().all()  # unseen -> missing, never a new level
    assert list(pre.transform_linear(rows).columns) == list(pre.transform_linear(rows.iloc[:5]).columns)


def test_errors_before_fit_or_on_empty_data(cfg):
    with pytest.raises(RuntimeError):
        FeaturePreprocessor(cfg).transform_linear(_training_rows())
    with pytest.raises(ValueError):
        FeaturePreprocessor(cfg).fit(_training_rows().iloc[:0])
    with pytest.raises(ValueError):
        FeaturePreprocessor(cfg).fit(pd.DataFrame({"x": [1]}))


def test_catalogue_matches_the_code():
    documented = set(feature_catalogue().index)
    produced = {*STATELESS_FEATURES, *LEARNED_NUMERIC, "n_unseen_categories"}
    assert documented == produced


def test_quality_report_flags_constants_and_duplicates():
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [5, 5, 5]})
    report = feature_quality_report(frame)
    assert report.loc["b", "duplicate_of_earlier_column"]
    assert report.loc["c", "constant"]
    assert not report.loc["a", ["constant", "duplicate_of_earlier_column"]].any()


def test_customer_join_and_tenure_check(cfg):
    raw = _raw().assign(customer_tenure_days=[10, 20, 50])
    customers = pd.DataFrame({"customer_id": ["A"], "signup_date": [pd.Timestamp("2025-12-22")], "city_tier": [2]})
    out = add_engineered_features(raw, CHARTS, cfg, customers)
    assert out["tenure_gap_days"].tolist() == [0, 0, 0]  # signup 10 days before first request, then +10, +40
    assert (out["city_tier"] == 2).all()


def test_validate_joins_reports_problems():
    requests = pd.DataFrame({"product_id": ["P1", "PX"], "customer_id": ["A", "A"]})
    products = pd.DataFrame({"product_id": ["P1", "P1"], "size_chart_id": ["SC1", None]})
    customers = pd.DataFrame({"customer_id": ["A"]})
    report = validate_joins(requests, products, customers, CHARTS)
    assert report["product_id_duplicates"] == 1
    assert report["requests_without_product"] == 1
    assert report["products_without_chart"] == 1


def test_categorical_dtype_columns_do_not_break_learned_features(cfg):
    """Regression: .map() on a categorical with distinct values returns a categorical that cannot be divided."""
    rows = _training_rows()
    for column in ("sub_category", "brand", "product_id"):
        rows[column] = pd.Categorical(rows[column])
    rows["sub_category"] = pd.Categorical([f"s{i % 7}" for i in range(len(rows))])
    rows["item_price_inr"] = np.arange(len(rows), dtype=float) + 100
    pre = FeaturePreprocessor(cfg).fit(rows)
    matrix = pre.transform_tree(rows)
    assert np.isfinite(matrix["price_vs_subcategory_median"]).all()
    assert pre.transform_linear(rows).isna().sum().sum() == 0
