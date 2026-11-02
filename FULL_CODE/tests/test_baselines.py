import numpy as np
import pandas as pd
import pytest

from r2sorter.baselines import (
    MODEL_ORDER,
    STOCHASTIC,
    abuse_metrics,
    build_model,
    business_cost,
    catboost_inputs,
    expected_calibration_error,
    predict_full,
    reliability_table,
)
from r2sorter.data import LABELS


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = (x["a"] + rng.normal(scale=0.5, size=n) > 1.0).astype(int)
    labels = np.where(y == 1, 0, rng.integers(1, 4, n))
    return x, labels


@pytest.mark.parametrize("name", MODEL_ORDER)
def test_every_baseline_returns_four_probability_columns(name):
    x, labels = _toy()
    model = build_model(name, 0).fit(x, labels)
    probs = predict_full(model, x)
    assert probs.shape == (len(x), len(LABELS))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_missing_class_is_padded_with_zero_probability():
    x, labels = _toy()
    labels = np.where(labels == 3, 2, labels)  # class 3 never seen
    probs = predict_full(build_model("logreg", 0).fit(x, labels), x)
    assert probs.shape[1] == 4 and (probs[:, 3] == 0).all()


def test_unknown_model_and_stochastic_set():
    with pytest.raises(KeyError):
        build_model("svm", 0)
    assert set(MODEL_ORDER) >= STOCHASTIC


def test_dummy_predicts_the_prior():
    x, labels = _toy()
    probs = predict_full(build_model("dummy", 0).fit(x, labels), x)
    assert np.allclose(probs[0], np.bincount(labels, minlength=4) / len(labels))


def test_ece_is_zero_for_a_perfectly_calibrated_constant():
    y = np.array([0, 0, 0, 1] * 50)
    assert expected_calibration_error(y, np.full(len(y), 0.25)) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(y, np.full(len(y), 0.9)) > 0.5


def test_metrics_and_reliability_shapes():
    y = np.array([0, 1] * 100)
    p = np.where(y == 1, 0.8, 0.2)
    metrics = abuse_metrics(y, p)
    assert metrics["abuse_AP"] == 1.0 and metrics["abuse_log_loss"] < 0.3
    assert len(reliability_table(y, p, bins=5)) == 5


def test_business_cost_uses_official_matrix(cfg):
    y = np.array([1, 1, 0, 0])
    sure_abuse = business_cost(np.array([1.0, 1.0, 0.0, 0.0]), y, cfg)
    assert sure_abuse["genuine_escalated"] == 0 and sure_abuse["abusive_approved"] == 0
    assert sure_abuse["saving_vs_approve_all"] == pytest.approx(1.0)
    wrong = business_cost(np.array([0.0, 0.0, 1.0, 1.0]), y, cfg)  # approves abusers, escalates honest customers
    assert wrong["genuine_escalated"] == 2 and wrong["abusive_approved"] == 2
    assert wrong["saving_vs_approve_all"] < 0


def test_catboost_inputs_turn_categories_into_strings_and_fit():
    x, labels = _toy()
    x["colour"] = pd.Categorical(np.where(x["a"] > 0, "red", "blue"))
    converted, names = catboost_inputs(x)
    assert names == ["colour"] and not isinstance(converted["colour"].dtype, pd.CategoricalDtype)
    model = build_model("catboost", 0, names).fit(converted, labels)
    assert predict_full(model, converted).shape == (len(x), 4)
