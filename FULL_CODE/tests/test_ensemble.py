import numpy as np
import pandas as pd
import pytest

from r2sorter.ensemble import blend, fit_simplex_weights, log_loss_of
from r2sorter.router import route
from r2sorter.tabpfn_runner import numeric_codes, stratified_subsample


def _noisy(y, noise, seed):
    rng = np.random.default_rng(seed)
    probs = np.full((len(y), 4), 0.05)
    probs[np.arange(len(y)), y] = 0.85
    probs = probs * (1 + noise * rng.random(probs.shape))
    return probs / probs.sum(axis=1, keepdims=True)


def test_blend_validates_and_normalises():
    a, b = np.eye(4)[[0, 1]] * 0.7 + 0.1, np.eye(4)[[1, 0]] * 0.7 + 0.1
    out = blend([a, b], np.array([0.5, 0.5]))
    assert np.allclose(out.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        blend([a, b], np.array([0.7, 0.7]))
    with pytest.raises(ValueError):
        blend([a, b], np.array([1.2, -0.2]))


def test_weights_prefer_the_better_member():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 600)
    good, bad = _noisy(y, 0.1, 1), _noisy(rng.permutation(y), 0.1, 2)  # second member is uninformative
    weights = fit_simplex_weights([good, bad], y)
    assert weights.sum() == pytest.approx(1.0) and (weights >= 0).all()
    assert weights[0] > 0.9
    assert log_loss_of(blend([good, bad], weights), y) <= log_loss_of(bad, y)


def test_single_member_weight_is_one():
    assert fit_simplex_weights([np.full((5, 4), 0.25)], np.zeros(5, dtype=int)).tolist() == [1.0]


def test_numeric_codes_and_subsample():
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": pd.Categorical(["x", "y", "x"])})
    coded, positions = numeric_codes(frame)
    assert positions == [1] and coded["c"].tolist() == [0.0, 1.0, 0.0]
    y = np.array([0] * 80 + [1] * 15 + [2] * 3 + [3] * 2)
    take = stratified_subsample(y, 50)
    assert set(y[take]) == {0, 1, 2, 3} and len(take) <= 60 and (np.diff(take) > 0).all()


def test_grid_bootstrap_and_cost_tuning():
    from r2sorter.ensemble import paired_bootstrap, row_costs, simplex_grid, tune_weights_by_cost

    grid = simplex_grid(3, 0.25)
    assert len(grid) == 15 and all(np.isclose(w.sum(), 1.0) for w in grid)
    mean, low, high = paired_bootstrap(np.full(200, 0.1))
    assert mean == pytest.approx(0.1) and low == pytest.approx(0.1) and high == pytest.approx(0.1)
    cfg = {
        "costs": {
            "genuine": {"auto_approve": 0, "inspect": 0.5, "escalate": 10},
            "abusive": {"auto_approve": 1, "inspect": 0.3, "escalate": 0},
        }
    }
    assert row_costs(np.array([0.0, 1.0]), np.array([0, 1]), cfg).tolist() == [0.0, 0.0]
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 400)
    good, bad = _noisy(y, 0.1, 3), _noisy(rng.permutation(y), 0.1, 4)
    weights, table = tune_weights_by_cost([good, bad], y, (y == 0).astype(int), cfg, step=0.25)
    assert weights.sum() == pytest.approx(1.0) and weights[0] >= weights[1] and len(table) == 5


def _toy_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "label": np.where(rng.random(n) < 0.1, "abusive", "genuine_fit_issue"),
            "customer_tenure_days": rng.integers(0, 400, n),
            "order_value_inr": rng.integers(200, 5000, n),
            "customer_reason": rng.choice(
                ["size chota hai", "too small", "wrong item received", "kapda kharab hai"], n
            ),
        }
    )


_CFG = {
    "costs": {
        "genuine": {"auto_approve": 0, "inspect": 0.5, "escalate": 10},
        "abusive": {"auto_approve": 1, "inspect": 0.3, "escalate": 0},
    },
    "fairness": {"new_customer_days": 90, "low_value_inr": 1000},
}


def test_fast_fairness_matches_the_audit_and_savings_match_business_cost():
    from r2sorter.baselines import business_cost
    from r2sorter.ensemble import fairness_masks, fast_official
    from r2sorter.fairness import audit

    frame = _toy_frame()
    y = (frame["label"] == "abusive").astype(int).to_numpy()
    p = np.clip(np.random.default_rng(1).random(len(frame)) * 0.5 + y * 0.3, 0, 1)
    masks = fairness_masks(frame, _CFG)
    result = fast_official(p, y, masks, _CFG)
    decisions = pd.Series(route(p, _CFG), index=frame.index)
    assert result["fairness"] == pytest.approx(audit(frame, decisions, _CFG)["fairness_score"])
    assert result["savings"] == pytest.approx(business_cost(p, y, _CFG)["saving_vs_approve_all"])
    assert result["primary"] == pytest.approx(
        0.7 * result["savings"] + 0.2 * result["fairness"] * min(1, max(0, result["savings"]) / 0.2)
    )


def test_multiplier_tuning_never_lowers_the_score_and_weights_stay_on_the_simplex():
    from r2sorter.ensemble import (
        apply_group_multipliers,
        fairness_masks,
        fast_official,
        tune_group_multipliers,
        tune_weights_by_official,
    )

    frame = _toy_frame(600, 3)
    y = (frame["label"] == "abusive").astype(int).to_numpy()
    p = np.clip(np.random.default_rng(2).random(len(frame)) * 0.4 + y * 0.4, 0, 1)
    masks = fairness_masks(frame, _CFG)
    before = fast_official(p, y, masks, _CFG)["primary"]
    multipliers, after = tune_group_multipliers(p, y, masks, _CFG)
    assert after >= before - 1e-12 and set(multipliers) == set(masks)
    assert apply_group_multipliers(p, masks, {k: 1.0 for k in masks}).tolist() == pytest.approx(p.tolist())
    y4 = np.where(y == 1, 0, 1)
    members = [_noisy(y4, 0.1, 5), _noisy(y4, 0.5, 6)]
    weights, table = tune_weights_by_official(members, y, masks, _CFG, step=0.5)
    assert weights.sum() == pytest.approx(1.0) and (weights >= 0).all() and len(table) == 3
