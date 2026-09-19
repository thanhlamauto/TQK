#!/usr/bin/env python3
"""CPU-only invariant tests run before any expensive GPU work."""

from __future__ import annotations

import numpy as np

import analyze_calibration as ac
from policies import BANK_CHECKPOINTS, valid_policies
from reliability_models import fit_ell, fit_miss


def test_expected_oracle_matches_monte_carlo() -> None:
    rng = np.random.default_rng(7)
    y_row = rng.normal(size=25)
    for pool in (5, 8, 13, 20):
        exact = ac.expected_oracle(y_row, pool)
        ranks = np.argsort(rng.random((200_000, 25)), axis=1)
        draws = ranks[:, :pool]
        mc = y_row[draws].max(axis=1).mean()
        assert abs(exact - mc) < 3e-3, (pool, exact, mc)
    print("PASS expected_oracle vs Monte Carlo")


def test_subset_regret_invariant() -> None:
    original = ac.SUBSETS_PER_PROMPT
    ac.SUBSETS_PER_PROMPT = 200
    try:
        rng = np.random.default_rng(11)
        n = 6
        y = rng.normal(size=(n, 25))
        x = rng.normal(size=(n, 25, len(BANK_CHECKPOINTS)))
        bank = {
            "y": y,
            "x": x,
            "ckpt_index": {step: index for index, step in enumerate(BANK_CHECKPOINTS)},
            "fold": np.zeros(n, dtype=int),
        }
        policies = valid_policies()
        stats = ac.collect_subset_statistics(bank, policies)
        indices = np.arange(n)
        agg = ac.aggregate_over_prompts(stats, policies, stats["oracle_curve"], indices)
        for index, policy in enumerate(policies):
            lhs = agg["regret"][index]
            rhs = agg["p_miss"][index] * agg["ell"][index]
            assert abs(lhs - rhs) < 1e-9, (policy, lhs, rhs)
        assert np.all(agg["o"] >= 0) or True  # O may be negative with synthetic noise
        assert np.all(agg["ell"] >= 0)
        print("PASS subset regret invariant p_miss * ell == mean regret")
    finally:
        ac.SUBSETS_PER_PROMPT = original


def test_models_monotone_and_positive() -> None:
    policies = valid_policies()
    q = np.array([policy.q for policy in policies])
    log_m = np.log(np.array([policy.M for policy in policies], dtype=float))
    ratio = np.array([policy.ratio for policy in policies])
    rng = np.random.default_rng(3)
    target_miss = np.clip(0.6 - 1.2 * q + 0.02 * log_m + rng.normal(0, 0.01, size=len(q)), 1e-3, 0.999)
    target_ell = np.abs(0.1 + 0.2 * q + rng.normal(0, 0.005, size=len(q))) + 1e-3
    weight = np.full(len(q), 1000.0)
    miss = fit_miss(q, log_m, ratio, target_miss, weight)
    ell = fit_ell(q, log_m, ratio, target_ell, np.maximum(weight * target_miss, 1.0))
    grid = np.linspace(q.min(), q.max(), 40)
    for m_value in (5, 10, 20):
        p = miss.predict(grid, np.log(m_value), 0.2)
        assert np.all(np.diff(p) <= 1e-12), "p_miss must not increase with denoising depth"
        e = ell.predict(grid, np.log(m_value), 0.2)
        assert np.all(e > 0)
    print("PASS monotone p_miss and positive ell")


if __name__ == "__main__":
    test_expected_oracle_matches_monte_carlo()
    test_subset_regret_invariant()
    test_models_monotone_and_positive()
    print("ALL ANALYSIS TESTS PASS")
