"""End-to-end pipeline: validate on a time-based holdout, then fit on all labelled data and write the submission."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from . import fairness, sizechart
from .calibration import AbuseCalibrator, expected_calibration_error
from .config import resolve_path
from .data import load_all
from .explain import justify_all
from .features import add_label_history, build_feature_table
from .model import StackModel, rolling_origin_predictions
from .router import break_even_thresholds, route
from .scoring import official_score

LOGGER = logging.getLogger(__name__)
PREDICTION_COLUMNS = ["return_id", "decision", "confidence", "justification"]


def predict_window(
    frame: pd.DataFrame,
    boundary: pd.Timestamp,
    target_mask: pd.Series,
    lag_days: int,
    cfg: dict[str, Any],
) -> tuple[StackModel, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit on labelled rows before ``boundary``, calibrate out-of-time, and score the ``target_mask`` rows."""
    seed = int(cfg["seed"])
    start = pd.Timestamp(cfg["validation"]["calibration_start"])
    out_of_time = rolling_origin_predictions(frame, start, boundary, cfg)
    calibrator = AbuseCalibrator(cfg["calibration"]["method"]).fit(
        out_of_time["p_abuse_raw"].to_numpy(),
        (frame.loc[out_of_time.index, "label"] == "abusive").to_numpy(),
    )
    known = add_label_history(frame, boundary, lag_days, seed)
    fit_rows = known[known["is_train"] & (known["return_request_date"] < boundary)]
    targets = known[target_mask]
    model = StackModel(cfg).fit(fit_rows)
    predicted = model.predict(targets)
    predicted["p_abuse"] = calibrator.transform(predicted["p_abuse_raw"].to_numpy())
    return model, targets, predicted, out_of_time


def build_predictions(
    model: StackModel, targets: pd.DataFrame, predicted: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Route every request and attach a confidence and a data-driven justification."""
    p_abuse = predicted["p_abuse"].to_numpy()
    decisions = route(p_abuse, cfg)
    contributions = model.contributions(targets)
    features = model.features(targets).join(targets[["cod_flag"]], rsuffix="_dup")
    justifications = justify_all(features, contributions, decisions, p_abuse, int(cfg["explain"]["top_reasons"]))
    return pd.DataFrame(
        {
            "return_id": targets["return_id"].to_numpy(),
            "decision": decisions,
            "confidence": np.round(p_abuse, 4),
            "justification": justifications,
        },
        index=targets.index,
    )


def validate(cfg: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    """Time-based holdout: nothing from the holdout months is used for fitting or calibration."""
    boundary = pd.Timestamp(cfg["validation"]["holdout_start"])
    labelled = frame["is_train"]
    mask = labelled & (frame["return_request_date"] >= boundary)
    lag = int((frame.loc[mask, "return_request_date"].max() - boundary).days) + 1
    model, targets, predicted, out_of_time = predict_window(frame, boundary, mask, lag, cfg)
    predictions = build_predictions(model, targets, predicted, cfg)
    result = official_score(predictions, targets)
    truth = (targets["label"] == "abusive").to_numpy()
    result["ece"] = expected_calibration_error(truth, predicted["p_abuse"].to_numpy())
    result["fairness_audit"] = fairness.audit(targets, predictions["decision"], cfg)
    result["decision_counts"] = predictions["decision"].value_counts().to_dict()
    _save_holdout_outputs(cfg, frame, targets, predicted, predictions, out_of_time)
    return result


def _save_holdout_outputs(
    cfg: dict[str, Any],
    frame: pd.DataFrame,
    targets: pd.DataFrame,
    predicted: pd.DataFrame,
    predictions: pd.DataFrame,
    out_of_time: pd.DataFrame,
) -> None:
    """Small CSVs so the phase notebooks can show calibration and fairness without retraining."""
    reports = resolve_path(cfg, "reports_dir")
    reports.mkdir(parents=True, exist_ok=True)
    columns = ["label", "customer_tenure_days", "order_value_inr", "customer_reason"]
    holdout = targets[columns].join(predicted[["p_abuse_raw", "p_abuse"]]).join(predictions[["decision"]])
    holdout.to_csv(reports / "holdout_predictions.csv", index_label="row")
    out = out_of_time[["p_abuse_raw"]].assign(abusive=(frame.loc[out_of_time.index, "label"] == "abusive").astype(int))
    out.to_csv(reports / "calibration_out_of_time.csv", index_label="row")


def write_submission(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit on all labelled data, predict the test period and write both submission files plus reports."""
    data = load_all(cfg)
    frame = build_feature_table(data["train"], data["test"], data["products"], cfg)
    test_start = data["test"]["return_request_date"].min()
    mask = ~frame["is_train"]
    model, targets, predicted, _ = predict_window(frame, test_start, mask, int(cfg["label_history_max_lag_days"]), cfg)
    predictions = build_predictions(model, targets, predicted, cfg)
    _check_submission(predictions, data["test"])
    submission_dir = resolve_path(cfg, "submission_dir")
    reports_dir = resolve_path(cfg, "reports_dir")
    for folder in (submission_dir, reports_dir):
        folder.mkdir(parents=True, exist_ok=True)
    predictions[PREDICTION_COLUMNS].to_csv(submission_dir / "R2_predictions.csv", index=False)

    table = sizechart.fit_return_table(data["train"], targets, predicted["p_genuine_fit_issue"], data["products"])
    top = sizechart.top_products(table, cfg)
    top[["rank", "product_id"]].to_csv(submission_dir / "R2_size_chart_top5.csv", index=False)
    table.sort_values("combined_fit_returns", ascending=False).head(15).to_csv(reports_dir / "size_chart_evidence.csv")
    model.feature_importance().head(25).to_csv(reports_dir / "feature_importance.csv", header=["gain_share"])
    summary = {
        "decision_counts": predictions["decision"].value_counts().to_dict(),
        "break_even": break_even_thresholds(cfg),
        "mean_confidence": float(predictions["confidence"].mean()),
    }
    (reports_dir / "submission_summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("wrote submission: %s", summary)
    return summary


def _check_submission(predictions: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if the file would be rejected by the scorer."""
    if set(predictions["return_id"]) != set(test["return_id"]) or predictions["return_id"].duplicated().any():
        raise ValueError("predictions must cover every test return_id exactly once")
    if not predictions["confidence"].between(0, 1).all():
        raise ValueError("confidence must lie in [0, 1]")
    if predictions["justification"].str.strip().eq("").any():
        raise ValueError("every row needs a justification")
