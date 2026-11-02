"""The Phase 5/6 evaluation protocol in one place, so later phases reuse exactly the same rows and features.

Phase 2 chronological 70/10/20 split: ``train`` fits models, ``calibration`` fits only the calibrator,
``test`` (the May-June validation period) is the common yardstick. The hidden July-September test is never touched.
Text scores for the train rows are rolling-origin out-of-fold (Phase 4);
calibration/test rows are scored by a model fitted on train.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import LABELS
from .features import build_feature_table, tabular_columns
from .nlp import TextCandidate, check_rolling, rolling_origin_out_of_fold
from .preprocess import FeaturePreprocessor, add_engineered_features
from .splits import chronological_split

PART_NAMES = ("train", "calibration", "test")
TEXT_COLUMNS = [f"txt_p_{label}" for label in LABELS]
REUSE_COLUMNS = ["hist_same_text", "hist_max_text_sim", "hist_n_similar_text"]


@dataclass
class Protocol:
    """Everything needed to build identical feature matrices for any model."""

    parts: dict[str, pd.DataFrame]
    preprocessor: FeaturePreprocessor
    text_scores: dict[str, np.ndarray]
    rule_columns: list[str]
    extra_scaler: StandardScaler
    rolling_check: dict[str, int]
    y_index: dict[str, np.ndarray] = field(default_factory=dict)
    y_abuse: dict[str, np.ndarray] = field(default_factory=dict)

    def matrix(self, kind: str, part: str) -> pd.DataFrame:
        """Feature arm C (structured + text scores + rule/shape + text reuse). kind: 'linear' or 'tree'."""
        rows = self.parts[part]
        base = self.preprocessor.transform_linear(rows) if kind == "linear" else self.preprocessor.transform_tree(rows)
        extra_columns = self.rule_columns + REUSE_COLUMNS
        extra = rows[extra_columns].reset_index(drop=True)
        if kind == "linear":
            extra = pd.DataFrame(self.extra_scaler.transform(extra), columns=extra_columns)
        text = pd.DataFrame(self.text_scores[part], columns=TEXT_COLUMNS)
        out = pd.concat([base.reset_index(drop=True), text, extra], axis=1)
        if len(out) != len(rows) or out.columns.duplicated().any():
            raise AssertionError("feature matrix misaligned with its rows")
        return out


def build_protocol(cfg: dict[str, Any], data: dict[str, pd.DataFrame], seed: int = 42) -> Protocol:
    """Build the split, the training-fitted preprocessing and the leakage-safe text scores."""
    train, test = data["train"], data["test"]
    frame = build_feature_table(train, test, data["products"], cfg)
    frame = add_engineered_features(frame, data["size_charts"], cfg, data["customers"])
    labelled = frame[frame["is_train"]].sort_values(["return_request_date", "return_id"]).reset_index(drop=True)
    plan = chronological_split(labelled["return_request_date"])
    parts = {name: labelled[plan == name].reset_index(drop=True) for name in PART_NAMES}
    if not (
        parts["train"].return_request_date.max() < parts["calibration"].return_request_date.min()
        and parts["calibration"].return_request_date.max() < parts["test"].return_request_date.min()
    ):
        raise AssertionError("splits must be strictly chronological")

    pre = FeaturePreprocessor(cfg).fit(parts["train"])
    params = {**cfg["text"], "svm_c": 0.5}
    text_model = TextCandidate("word", "logreg", params, seed)
    train_rows = parts["train"]
    oof, kind, fit_end, _ = rolling_origin_out_of_fold(
        text_model,
        train_rows["customer_reason"],
        train_rows["label"],
        train_rows["customer_id"],
        train_rows["return_request_date"],
        60,
        int(cfg["validation"]["text_oof_folds"]),
    )
    rolling_check = check_rolling(kind, fit_end, train_rows["return_request_date"])
    fitted = text_model.fit(train_rows["customer_reason"], train_rows["label"])
    text_scores = {"train": oof, **{n: fitted.scores(parts[n]["customer_reason"]) for n in ("calibration", "test")}}

    rules = [c for c in tabular_columns(frame) if c.startswith(("ev_", "txt_len", "txt_n_", "txt_upper", "txt_has"))]
    scaler = StandardScaler().fit(parts["train"][rules + REUSE_COLUMNS])
    protocol = Protocol(parts, pre, text_scores, rules, scaler, rolling_check)
    for name, rows in parts.items():
        protocol.y_index[name] = rows["label"].map(LABELS.index).to_numpy()
        protocol.y_abuse[name] = (protocol.y_index[name] == LABELS.index("abusive")).astype(int)
    return protocol
