#!/usr/bin/env python3
"""CPU-only invariant tests for Adaptive-K (run before any GPU work)."""

from __future__ import annotations

import itertools

import numpy as np

from regret_models import (
    FEATURES,
    build_features,
    exact_budget_allocation,
    project_monotone,
)


def brute_force(costs: np.ndarray) -> float:
    n = costs.shape[0]
    best = np.inf
    for choice in itertools.product((1, 2, 3), repeat=n):
        if sum(choice) != 2 * n:
            continue
        total = sum(costs[p, choice[p] - 1] for p in range(n))
        best = min(best, total)
    return best


def test_allocation_exact_and_balanced() -> None:
    rng = np.random.default_rng(0)
    for n in (4, 5, 6):
        for _ in range(20):
            raw = rng.random((n, 3)) * 2.0
            costs = np.sort(raw, axis=1)[:, ::-1]  # monotone decreasing, like regrets
            chosen = exact_budget_allocation(costs)
            assert chosen.sum() == 2 * n
            assert np.bincount(chosen, minlength=4)[1] == np.bincount(chosen, minlength=4)[3]
            dp_cost = sum(costs[p, chosen[p] - 1] for p in range(n))
            assert abs(dp_cost - brute_force(costs)) < 1e-9, (n, dp_cost, brute_force(costs))
    print("PASS exact-budget allocation matches brute force and n1 == n3")


def test_projection_monotone_nonnegative() -> None:
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(1000, 3)) * 2.0
    projected = project_monotone(raw)
    assert np.all(projected >= -1e-12)
    assert np.all(projected[:, 0] >= projected[:, 1] - 1e-9)
    assert np.all(projected[:, 1] >= projected[:, 2] - 1e-9)
    print("PASS monotone projection")


def test_features_finite() -> None:
    rng = np.random.default_rng(2)
    scores = -np.sort(-rng.normal(size=(50, 10)), axis=1)  # descending
    features = build_features(scores)
    assert features.shape == (50, len(FEATURES))
    assert np.isfinite(features).all()
    g1 = features[:, FEATURES.index("g1")]
    assert np.all(g1 >= -1e-9)
    print("PASS feature construction")


if __name__ == "__main__":
    test_allocation_exact_and_balanced()
    test_projection_monotone_nonnegative()
    test_features_finite()
    print("ALL ADAPTIVE-K TESTS PASS")
