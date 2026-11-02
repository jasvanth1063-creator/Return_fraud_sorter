"""Two-stage model: a text classifier feeds its (out-of-fold) probabilities into gradient boosting."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import LABELS
from .features import add_label_history, tabular_columns
from .text import TextReasonModel, out_of_fold_text_probs, text_probability_frame

LOGGER = logging.getLogger(__name__)
ABUSIVE_INDEX = LABELS.index("abusive")


class StackModel:
    """Predicts P(abusive) and the four-way return reason for each request."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._seed = int(cfg["seed"])
        self.columns: list[str] = []
        self._text_model: TextReasonModel | None = None
        self._abuse: list[lgb.LGBMClassifier] = []
        self._reason: lgb.LGBMClassifier | None = None

    def fit(self, rows: pd.DataFrame) -> StackModel:
        """Fit on labelled rows only. Text probabilities for these rows are out-of-fold (by customer)."""
        self.columns = tabular_columns(rows)
        folds = int(self._cfg["validation"]["text_oof_folds"])
        oof = out_of_fold_text_probs(
            rows["customer_reason"],
            rows["label"],
            rows["customer_id"],
            self._cfg["text"],
            folds,
            self._seed,
        )
        self._text_model = TextReasonModel(self._cfg["text"], self._seed).fit(rows["customer_reason"], rows["label"])
        features = pd.concat([rows[self.columns], text_probability_frame(oof, rows.index)], axis=1)
        reason_codes = rows["label"].map(LABELS.index)
        target = (reason_codes == ABUSIVE_INDEX).astype(int)
        self._abuse = [
            lgb.LGBMClassifier(
                objective="binary",
                random_state=self._seed + offset,
                verbose=-1,
                **self._cfg["model"]["abuse"],
            ).fit(features, target)
            for offset in range(int(self._cfg["model"]["n_seeds"]))
        ]
        self._reason = lgb.LGBMClassifier(
            objective="multiclass",
            random_state=self._seed,
            verbose=-1,
            **self._cfg["model"]["reason"],
        ).fit(features, reason_codes)
        LOGGER.info("fitted stack model on %d rows with %d features", len(rows), features.shape[1])
        return self

    def _features(self, rows: pd.DataFrame) -> pd.DataFrame:
        if self._text_model is None:
            raise RuntimeError("StackModel must be fitted first")
        probs = self._text_model.predict_proba(rows["customer_reason"])
        return pd.concat([rows[self.columns], text_probability_frame(probs, rows.index)], axis=1)

    def features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Model input (tabular + text probabilities) for the given rows."""
        return self._features(rows)

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Raw (uncalibrated) P(abusive) and four-way reason probabilities."""
        if not self._abuse or self._reason is None:
            raise RuntimeError("StackModel must be fitted first")
        features = self._features(rows)
        p_abuse = np.mean([model.predict_proba(features)[:, 1] for model in self._abuse], axis=0)
        out = pd.DataFrame({"p_abuse_raw": p_abuse}, index=rows.index)
        reason = self._reason.predict_proba(features)
        for position, label in enumerate(LABELS):
            out[f"p_{label}"] = reason[:, position]
        return out

    def contributions(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Per-feature log-odds contribution to the abuse prediction (SHAP-style, from LightGBM)."""
        if not self._abuse:
            raise RuntimeError("StackModel must be fitted first")
        features = self._features(rows)
        raw = np.mean([m.booster_.predict(features, pred_contrib=True) for m in self._abuse], axis=0)
        return pd.DataFrame(raw[:, :-1], index=rows.index, columns=features.columns)

    def feature_importance(self) -> pd.Series:
        """Share of total gain per feature for the abuse model."""
        if not self._abuse:
            raise RuntimeError("StackModel must be fitted first")
        booster = self._abuse[0].booster_
        gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
        return (gain / gain.sum()).sort_values(ascending=False)


def rolling_origin_predictions(
    frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Out-of-time predictions: for each month in [start, end) fit on everything BEFORE it, predict that month."""
    pieces = []
    for month_start in pd.date_range(start, end, freq="MS", inclusive="left"):
        month_end = month_start + pd.offsets.MonthBegin(1)
        known = add_label_history(frame, month_start, int(cfg["label_history_max_lag_days"]), int(cfg["seed"]))
        labelled = known[known["is_train"]]
        fit_rows = labelled[labelled["return_request_date"] < month_start]
        month_rows = labelled[
            (labelled["return_request_date"] >= month_start) & (labelled["return_request_date"] < month_end)
        ]
        if month_rows.empty:
            continue
        LOGGER.info(
            "rolling fold %s: fit %d rows, predict %d",
            month_start.date(),
            len(fit_rows),
            len(month_rows),
        )
        pieces.append(StackModel(cfg).fit(fit_rows).predict(month_rows))
    return pd.concat(pieces) if pieces else pd.DataFrame()


def months_between(first: pd.Timestamp, last: pd.Timestamp) -> int:
    """Number of whole months between two timestamps."""
    return int(np.floor((last - first).days / 30.4))
