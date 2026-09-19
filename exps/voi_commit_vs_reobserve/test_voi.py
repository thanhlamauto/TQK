#!/usr/bin/env python3
"""CPU-only invariant tests for the VOI action family."""

from __future__ import annotations

import numpy as np

from simulate_pools import simulate
from voi_models import ACTION_A_NFE, ACTION_B_NFE, PSP_NFE, FEATURES, build_features


def brute_force_pool(scores16, scores32, finals, pool):
    order = sorted(pool.tolist(), key=lambda i: (-scores16[i], i))
    top2 = order[:2]
    y_a = max(finals[i] for i in top2)
    top4 = order[:4]
    best = sorted(top4, key=lambda i: (-scores32[i], i))[0]
    y_b = finals[best]
    psp_pool = pool[:8]
    psp_order = sorted(psp_pool.tolist(), key=lambda i: (-scores16[i], i))[:4]
    psp_survivors = sorted(psp_order, key=lambda i: (-scores32[i], i))[:2]
    y_psp = max(finals[i] for i in psp_survivors)
    return y_a, y_b, y_psp


def test_nfe() -> None:
    assert ACTION_A_NFE == 10 * 16 + 2 * 48 == 256
    assert ACTION_B_NFE == 10 * 16 + 4 * 16 + 1 * 32 == 256
    assert PSP_NFE == 8 * 16 + 4 * 16 + 2 * 32 == 256
    print("PASS NFE accounting (A=B=PSP=256)")


def test_simulate_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        scores16 = rng.normal(size=25)
        scores32 = rng.normal(size=25)
        finals = rng.normal(size=25)
        pools = np.argsort(rng.random((50, 25)), axis=1)[:, :10]
        y_a, y_b, y_psp, a_ids, b_winner = simulate(scores16, scores32, finals, pools)
        for row in range(50):
            ea, eb, ep = brute_force_pool(scores16, scores32, finals, pools[row])
            assert abs(y_a[row] - ea) < 1e-12
            assert abs(y_b[row] - eb) < 1e-12
            assert abs(y_psp[row] - ep) < 1e-12
            assert set(a_ids[row].tolist()) == set(
                sorted(pools[row].tolist(), key=lambda i: (-scores16[i], i))[:2]
            )
    print("PASS simulate matches brute force for A, B and PSP")


def test_features() -> None:
    rng = np.random.default_rng(1)
    scores = -np.sort(-rng.normal(size=(100, 10)), axis=1)
    features = build_features(scores)
    assert features.shape == (100, len(FEATURES))
    assert np.isfinite(features).all()
    assert np.all(features[:, FEATURES.index("g1")] >= -1e-9)
    print("PASS feature construction")


if __name__ == "__main__":
    test_nfe()
    test_simulate_matches_brute_force()
    test_features()
    print("ALL VOI TESTS PASS")
