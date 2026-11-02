import pandas as pd
import pytest

from r2sorter.splits import assert_chronological, chronological_split, describe_splits


def _frame(days: int = 100, per_day: int = 3) -> pd.DataFrame:
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta([d for d in range(days) for _ in range(per_day)], unit="D")
    return pd.DataFrame(
        {
            "return_request_date": dates,
            "label": (["abusive", "genuine_fit_issue", "abusive"] * days * per_day)[: len(dates)],
        }
    )


def test_split_proportions_and_order():
    frame = _frame()
    assignment = chronological_split(frame["return_request_date"])
    shares = assignment.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.7, abs=0.02)
    assert shares["calibration"] == pytest.approx(0.1, abs=0.02)
    assert shares["test"] == pytest.approx(0.2, abs=0.02)
    assert_chronological(frame, assignment, ["train", "calibration", "test"])


def test_a_calendar_day_is_never_split():
    frame = _frame(days=50, per_day=7)
    assignment = chronological_split(frame["return_request_date"])
    per_day = assignment.groupby(frame["return_request_date"]).nunique()
    assert (per_day == 1).all()


def test_overlapping_splits_are_rejected():
    frame = _frame(days=10)
    shuffled = pd.Series(["train", "test"] * (len(frame) // 2), index=frame.index)
    with pytest.raises(AssertionError):
        assert_chronological(frame, shuffled, ["train", "test"])


def test_bad_fractions_and_missing_dates_are_rejected():
    frame = _frame(days=5)
    with pytest.raises(ValueError):
        chronological_split(frame["return_request_date"], fractions=(0.5, 0.2, 0.2))
    with pytest.raises(ValueError):
        chronological_split(pd.Series([pd.NaT, pd.Timestamp("2026-01-01")]))


def test_describe_reports_counts_and_label_mix():
    frame = _frame()
    table = describe_splits(frame, chronological_split(frame["return_request_date"]))
    assert table["rows"].sum() == len(frame)
    assert table.loc["train", "abusive"] == pytest.approx(2 / 3, abs=0.01)
