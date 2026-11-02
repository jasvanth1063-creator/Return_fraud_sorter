"""Run the OFFICIAL scorer (tools/evaluate.py, unmodified copy from the problem pack) on in-memory frames."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from .config import PROJECT_ROOT

OFFICIAL_SCORER_PATH = PROJECT_ROOT / "tools" / "evaluate.py"
_TRUTH_COLUMNS = [
    "return_id",
    "label",
    "customer_tenure_days",
    "order_value_inr",
    "customer_reason",
]


def load_official_scorer() -> ModuleType:
    """Import tools/evaluate.py by path (it is not an installed package)."""
    spec = importlib.util.spec_from_file_location("official_evaluate", OFFICIAL_SCORER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the official scorer from {OFFICIAL_SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_score(predictions: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    """Score predictions (return_id, decision, confidence, justification) against labelled rows."""
    scorer = load_official_scorer()
    with tempfile.TemporaryDirectory() as tmp:
        pred_path = Path(tmp) / "predictions.csv"
        truth_path = Path(tmp) / "truth.csv"
        predictions[["return_id", "decision", "confidence", "justification"]].to_csv(pred_path, index=False)
        truth[_TRUTH_COLUMNS].to_csv(truth_path, index=False)
        return scorer.score(pred_path, truth_path, None, None)
