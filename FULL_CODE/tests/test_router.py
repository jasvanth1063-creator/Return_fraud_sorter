import numpy as np
import pytest

from r2sorter.router import break_even_thresholds, expected_costs, route


def test_break_even_matches_official_cost_matrix(cfg):
    thresholds = break_even_thresholds(cfg)
    assert thresholds["approve_to_inspect"] == pytest.approx(0.5 / 1.2)
    assert thresholds["inspect_to_escalate"] == pytest.approx(9.5 / 9.8)


def test_route_follows_expected_cost(cfg):
    p = np.array([0.0, 0.3, 0.5, 0.96, 0.99])
    assert list(route(p, cfg)) == ["auto_approve", "auto_approve", "inspect", "inspect", "escalate"]


def test_expected_cost_is_probability_weighted(cfg):
    costs = expected_costs(np.array([0.5]), cfg)[0]
    assert costs[0] == pytest.approx(0.5)  # approve: 0.5*0 + 0.5*1
    assert costs[2] == pytest.approx(5.0)  # escalate: 0.5*10 + 0.5*0


def test_route_edge_probabilities(cfg):
    assert list(route(np.array([0.0, 1.0]), cfg)) == ["auto_approve", "escalate"]
