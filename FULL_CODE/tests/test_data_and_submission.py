import pandas as pd
import pytest

from r2sorter.data import DataError, read_requests
from r2sorter.pipeline import _check_submission


def test_read_requests_rejects_missing_file(tmp_path):
    with pytest.raises((DataError, FileNotFoundError)):
        read_requests(tmp_path / "nope.csv", with_label=False)


def test_read_requests_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"return_id": ["R1"]}).to_csv(path, index=False)
    with pytest.raises(DataError):
        read_requests(path, with_label=False)


def _predictions(ids, confidence=0.1, justification="ok"):
    return pd.DataFrame(
        {
            "return_id": ids,
            "decision": "auto_approve",
            "confidence": confidence,
            "justification": justification,
        }
    )


def test_check_submission_accepts_exact_cover():
    test = pd.DataFrame({"return_id": ["a", "b"]})
    _check_submission(_predictions(["a", "b"]), test)


@pytest.mark.parametrize("ids", [["a"], ["a", "a"], ["a", "b", "c"]])
def test_check_submission_rejects_missing_duplicate_or_extra_ids(ids):
    with pytest.raises(ValueError):
        _check_submission(_predictions(ids), pd.DataFrame({"return_id": ["a", "b"]}))


def test_check_submission_rejects_bad_confidence_and_blank_justification():
    test = pd.DataFrame({"return_id": ["a", "b"]})
    with pytest.raises(ValueError):
        _check_submission(_predictions(["a", "b"], confidence=1.5), test)
    with pytest.raises(ValueError):
        _check_submission(_predictions(["a", "b"], justification=" "), test)
