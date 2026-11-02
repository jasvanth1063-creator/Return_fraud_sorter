"""Shared fixtures: tiny hand-made frames, so tests run in seconds and never need the real data."""

from __future__ import annotations

import pandas as pd
import pytest

from r2sorter.config import load_config


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture()
def small_frame() -> pd.DataFrame:
    """Two customers over six days; rows already sorted by date as the pipeline guarantees."""
    dates = pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-05", "2026-01-06", "2026-01-09", "2026-01-10"])
    frame = pd.DataFrame(
        {
            "return_id": [f"RET-{i}" for i in range(6)],
            "customer_id": ["A", "A", "B", "A", "A", "B"],
            "return_request_date": dates,
            "label": ["abusive", "abusive", "genuine_fit_issue", None, None, None],
            "is_train": [True, True, True, False, False, False],
        }
    )
    frame["day"] = (frame["return_request_date"] - dates[0]).dt.days
    return frame
