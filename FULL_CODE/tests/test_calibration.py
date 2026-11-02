import numpy as np
import pytest

from r2sorter.calibration import AbuseCalibrator, expected_calibration_error


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibrator_is_monotone_and_bounded(method):
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 2000)
    truth = rng.uniform(0, 1, 2000) < raw**2
    calibrated = AbuseCalibrator(method).fit(raw, truth).transform(np.linspace(0, 1, 50))
    assert calibrated.min() >= 0.001 and calibrated.max() <= 0.999
    assert np.all(np.diff(calibrated) >= -1e-9)


def test_calibrator_fixes_overconfident_scores():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, 5000)
    truth = rng.uniform(0, 1, 5000) < raw * 0.3
    fixed = AbuseCalibrator("isotonic").fit(raw, truth).transform(raw)
    assert expected_calibration_error(truth, fixed) < expected_calibration_error(truth, raw)


def test_calibrator_rejects_empty_or_mismatched_input():
    with pytest.raises(ValueError):
        AbuseCalibrator().fit(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        AbuseCalibrator().fit(np.array([0.1, 0.2]), np.array([1]))
    with pytest.raises(ValueError):
        AbuseCalibrator("magic")


def test_perfectly_calibrated_ece_is_zero():
    assert expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
