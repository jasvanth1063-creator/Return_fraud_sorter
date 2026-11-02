"""Chronological splits and the checks that prove they are leakage-free."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .data import LABELS

PLAN_FRACTIONS = (0.70, 0.10, 0.20)
PLAN_NAMES = ("train", "calibration", "test")


def chronological_split(
    dates: pd.Series,
    fractions: Sequence[float] = PLAN_FRACTIONS,
    names: Sequence[str] = PLAN_NAMES,
) -> pd.Series:
    """Assign each row to a split by request date; a calendar day is never divided between two splits."""
    if len(fractions) != len(names) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must match names and sum to 1")
    if dates.isna().any() or len(dates) == 0:
        raise ValueError("dates must be non-empty and have no missing values")
    per_day = dates.dt.normalize().value_counts().sort_index()
    cumulative = per_day.cumsum() / per_day.sum()
    assignment = pd.Series(names[-1], index=dates.index, dtype="object")
    day = dates.dt.normalize()
    start = pd.Timestamp.min
    for name, upper in zip(names[:-1], np.cumsum(fractions)[:-1], strict=True):
        last_day = cumulative.index[np.argmax(cumulative.to_numpy() >= upper - 1e-12)]
        assignment[(day > start) & (day <= last_day)] = name
        start = last_day
    return assignment


def describe_splits(
    frame: pd.DataFrame, assignment: pd.Series, date_column: str = "return_request_date"
) -> pd.DataFrame:
    """First/last date, row count, share and label mix of each split."""
    rows = []
    for name in pd.unique(assignment):
        part = frame[assignment == name]
        record = {
            "split": name,
            "first_date": part[date_column].min().date(),
            "last_date": part[date_column].max().date(),
            "rows": len(part),
            "share_of_rows": round(len(part) / len(frame), 3),
            "days": part[date_column].dt.normalize().nunique(),
        }
        if "label" in part and part["label"].notna().any():
            shares = part["label"].value_counts(normalize=True)
            record.update({f"{label}": round(float(shares.get(label, 0.0)), 3) for label in LABELS})
        rows.append(record)
    return pd.DataFrame(rows).set_index("split")


def assert_chronological(
    frame: pd.DataFrame,
    assignment: pd.Series,
    order: Sequence[str],
    date_column: str = "return_request_date",
) -> None:
    """Every split must end strictly before the next one starts, and the splits must not overlap."""
    last_seen = pd.Timestamp.min
    for name in order:
        part = frame.loc[assignment == name, date_column]
        if part.empty:
            raise AssertionError(f"split {name!r} is empty")
        if part.min() <= last_seen:
            raise AssertionError(f"split {name!r} starts on {part.min().date()}, not after {last_seen.date()}")
        last_seen = part.max()
    if (~assignment.isin(order)).any():
        raise AssertionError("some rows belong to no split")
