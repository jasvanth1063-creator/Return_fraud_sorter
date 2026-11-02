import numpy as np
import pandas as pd
import pytest

from r2sorter.advanced import class_weights, fit_lgbm, importance_table, lgbm_params, measure
from r2sorter.data import LABELS


def _toy(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = np.where(x["a"] > 1.2, 0, rng.integers(1, 4, n))
    return x, y


def test_class_weights_are_mild_and_largest_class_is_one():
    weights = class_weights(np.array([0] * 8 + [1] * 92))
    assert weights[1] == 1.0 and 1.0 < weights[0] < 92 / 8  # square root, not full inverse frequency


@pytest.mark.parametrize("variant", ["plain", "weighted", "robust"])
def test_variants_fit_and_predict_four_probabilities(variant):
    x, y = _toy()
    model = fit_lgbm(variant, 0, x, y)
    probs = model.predict_proba(x)
    assert probs.shape == (len(x), len(LABELS)) and np.allclose(probs.sum(axis=1), 1.0)


def test_robust_is_more_regularised_than_plain():
    plain, robust = lgbm_params("plain", 0), lgbm_params("robust", 0)
    assert robust["reg_lambda"] > plain["reg_lambda"] and robust["min_child_samples"] > plain["min_child_samples"]
    with pytest.raises(KeyError):
        lgbm_params("nope", 0)


def test_measure_and_importance():
    x, y = _toy()
    probs, cost = measure(lambda: fit_lgbm("plain", 0, x, y), lambda m: m.predict_proba(x), len(x))
    assert probs.shape[0] == len(x) and cost["train_seconds"] >= 0 and cost["inference_ms_per_1k_rows"] >= 0
    table = importance_table(fit_lgbm("plain", 0, x, y), list(x.columns), top=3)
    assert len(table) == 3 and table["gain_share"].iloc[0] >= table["gain_share"].iloc[-1]
    assert table.index[0] == "a"  # the only informative column
