import pandas as pd

from r2sorter import fairness, sizechart


def _frame():
    return pd.DataFrame(
        {
            "label": ["genuine_fit_issue"] * 6 + ["abusive"],
            "customer_tenure_days": [10, 20, 400, 500, 600, 700, 5],
            "order_value_inr": [500, 2000, 2000, 2000, 500, 3000, 100],
            "customer_reason": [
                "kapda kharab hai",
                "size small",
                "size small",
                "tight",
                "too big",
                "loose",
                "x",
            ],
        }
    )


def test_audit_ignores_abusive_rows_and_scores_equal_treatment_as_one(cfg):
    frame = _frame()
    decisions = pd.Series(["auto_approve"] * 6 + ["escalate"])
    result = fairness.audit(frame, decisions, cfg)
    assert result["fairness_score"] == 1.0
    assert result["new_vs_established"]["ratio"] == 1.0


def test_audit_penalises_flagging_new_customers_more(cfg):
    frame = _frame()
    decisions = pd.Series(
        [
            "inspect",
            "inspect",
            "auto_approve",
            "auto_approve",
            "auto_approve",
            "auto_approve",
            "auto_approve",
        ]
    )
    result = fairness.audit(frame, decisions, cfg)
    assert result["new_vs_established"]["ratio"] > 10
    assert result["new_vs_established"]["pair_score"] == 0.0


def test_top_products_ranks_by_fit_volume_and_respects_minimum(cfg):
    table = pd.DataFrame(
        {
            "combined_fit_returns": [50, 40, 30, 20, 10, 9, 99],
            "train_returns": [60, 60, 60, 60, 60, 60, 5],
            "test_returns": [10] * 6 + [1],
        },
        index=[f"P{i}" for i in range(7)],
    )
    top = sizechart.top_products(table, cfg)
    assert list(top["product_id"]) == ["P0", "P1", "P2", "P3", "P4"]
    assert list(top["rank"]) == [1, 2, 3, 4, 5]
