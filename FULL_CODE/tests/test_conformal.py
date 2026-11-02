import numpy as np
import pytest

from r2sorter.conformal import (
    conformal_quantile,
    cross_fit_calibrated,
    ensure_nonempty,
    fit_thresholds,
    prediction_sets,
    rescale_four_class,
    set_labels,
    wilson_interval,
)


def _data(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    raw = rng.dirichlet(np.ones(4) * 0.8, n)
    raw[np.arange(n), y] += 0.8  # informative but noisy
    return raw / raw.sum(axis=1, keepdims=True), y


def test_rescale_rows_sum_to_one_and_set_abusive():
    probs, _ = _data(50, 0)
    new = rescale_four_class(probs, np.full(50, 0.2))
    assert np.allclose(new.sum(axis=1), 1.0) and np.allclose(new[:, 0], 0.2)
    ratio_before = probs[:, 1] / probs[:, 2]
    assert np.allclose(new[:, 1] / new[:, 2], ratio_before)


def test_quantile_finite_sample_rule():
    scores = np.arange(1, 10) / 10
    assert conformal_quantile(scores, 0.2) == pytest.approx(0.8)  # ceil(10 * 0.8) = 8th smallest of 9 scores
    assert conformal_quantile(np.array([0.1, 0.2]), 0.01) == float("inf")


def test_marginal_and_mondrian_coverage_on_exchangeable_data():
    cal_p, cal_y = _data(6000, 1)
    test_p, test_y = _data(6000, 2)
    for mondrian in (False, True):
        sets = prediction_sets(test_p, fit_thresholds(cal_p, cal_y, 0.1, mondrian))
        assert sets[np.arange(len(test_y)), test_y].mean() >= 0.88
    mond = prediction_sets(test_p, fit_thresholds(cal_p, cal_y, 0.1, True))
    for k in range(4):
        assert mond[test_y == k, k].mean() >= 0.86


def test_nonempty_fix_labels_and_cross_fit():
    probs = np.array([[0.4, 0.3, 0.2, 0.1]])
    fixed, share = ensure_nonempty(np.zeros((1, 4), dtype=bool), probs)
    assert share == 1.0 and fixed.tolist() == [[True, False, False, False]]
    assert set_labels(np.array([[True, False, True, False]])) == ["{abusive, genuine_fit_issue}"]
    rng = np.random.default_rng(0)
    raw = rng.random(500)
    truth = (rng.random(500) < raw).astype(int)
    out = cross_fit_calibrated(raw, truth, "sigmoid", folds=5)
    assert out.shape == (500,) and ((out >= 0) & (out <= 1)).all()
    low, high = wilson_interval(90, 100)
    assert 0.8 < low < 0.9 < high < 0.96


def test_bagged_isotonic_is_smoother_and_valid():
    from r2sorter.calibration import AbuseCalibrator

    rng = np.random.default_rng(0)
    raw = rng.random(600)
    truth = (rng.random(600) < raw).astype(int)
    plain = AbuseCalibrator("isotonic").fit(raw, truth)
    bagged = AbuseCalibrator("bagged_isotonic", n_bags=20).fit(raw, truth)
    grid = np.linspace(0.01, 0.99, 400)
    assert len(np.unique(bagged.transform(grid))) > len(np.unique(plain.transform(grid)))
    assert ((bagged.transform(grid) >= 0.001) & (bagged.transform(grid) <= 0.999)).all()


def test_recency_weighted_thresholds_and_language_groups():
    import pandas as pd

    from r2sorter.conformal import fit_thresholds_weighted, recency_weights, weighted_conformal_quantile
    from r2sorter.nlp import language_group3

    dates = pd.Series(pd.to_datetime(["2026-04-01", "2026-04-15", "2026-04-30"]))
    weights = recency_weights(dates, 14)
    assert weights[-1] == 1.0 and weights[0] < weights[1] < weights[2]
    scores = np.arange(1, 101) / 100
    flat = weighted_conformal_quantile(scores, np.ones(100), 0.1)
    assert flat == pytest.approx(0.9, abs=0.011)
    assert weighted_conformal_quantile(np.array([]), np.array([]), 0.1) == float("inf")
    probs, y = _data(300, 4)
    assert fit_thresholds_weighted(probs, y, np.ones(300), 0.1, True).shape == (4,)
    groups = language_group3(
        pd.Series(["size too small", "size bahut chota hai", "kapda kharab hai nahi pasand", None])
    )
    assert groups.iloc[0] == "english_like" and set(groups) <= {"english_like", "hinglish_like", "hindi_dominant"}
