#!/usr/bin/env python3
"""CPU-only invariant tests for predictor v2 features and objective."""

from __future__ import annotations

import numpy as np

from voi_v2_models import (
    F0_FEATURES,
    F1_EXTRA,
    _kendall_tau,
    _rank_rows,
    _top4_churn,
    build_f0,
    build_f1,
    cap_weights,
)


def test_f0_shape() -> None:
    rng = np.random.default_rng(0)
    scores = -np.sort(-rng.normal(size=(200, 10)), axis=1)
    f0 = build_f0(scores)
    assert f0.shape == (200, len(F0_FEATURES))
    assert np.isfinite(f0).all()
    print("PASS F0")


def test_f1_shapes_and_semantics() -> None:
    rng = np.random.default_rng(1)
    ids = np.tile(np.arange(10), (300, 1))
    s12 = rng.normal(size=(300, 10))
    s14 = s12 + 0.1 * rng.normal(size=(300, 10))
    s16 = s14 + 0.1 * rng.normal(size=(300, 10))
    f1 = build_f1(s12, s14, s16, ids)
    assert f1.shape == (300, len(F1_EXTRA))
    assert np.isfinite(f1).all()
    # Perfect monotone agreement -> tau = 1.
    same = np.tile(np.arange(10.0), (5, 1))
    assert np.allclose(_kendall_tau(same, same), 1.0)
    assert np.allclose(_kendall_tau(same, same[:, ::-1]), -1.0)
    # Churn: identical sets -> 0, disjoint -> 1.
    a = np.tile(np.arange(4), (2, 1))
    assert np.allclose(_top4_churn(a, a), 0.0)
    assert np.allclose(_top4_churn(a, a + 5), 1.0)
    # m_rescue defined and finite.
    assert np.isfinite(f1[:, F1_EXTRA.index("m_rescue")]).all()
    print("PASS F1 and vectorised tau/churn")


def test_cap_weights() -> None:
    values = np.concatenate([np.full(95, 1.0), np.full(5, 100.0)])
    weights = cap_weights(values, 0.95)
    cap = float(np.quantile(values, 0.95))
    assert weights.max() <= cap + 1e-9
    assert weights[values == 100.0].max() <= cap + 1e-9
    print("PASS weight capping")


if __name__ == "__main__":
    test_f0_shape()
    test_f1_shapes_and_semantics()
    test_cap_weights()
    print("ALL VOI V2 TESTS PASS")
