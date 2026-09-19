"""Regret predictors and the exact-budget allocator for Adaptive-K.

Only step-16 information is used as policy input.  Features are built from the
10 step-16 ImageReward scores of a candidate pool, sorted descending with
candidate-id tie-breaking::

    s1 >= s2 >= ... >= s10

    median_s = median(s)
    MAD_s    = median(|s - median_s|) + eps
    z_i      = (s_i - median_s) / MAD_s

    g1 = z1 - z2
    g2 = z2 - z3
    g3 = z3 - z4

The primary policy predictor is Model B (gradient-boosted trees, one regressor
per K) followed by a monotone projection ``Lhat1 >= Lhat2 >= Lhat3 >= 0``.  Model
A (1-D isotonic regression on the boundary gap ``g_K``) is reported as the
low-capacity baseline.  The choice of Model B as primary was made before any
calibration label was inspected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression

EPS = 1e-6

FEATURES = (
    "g1",
    "g2",
    "g3",
    "z1",
    "z2",
    "z3",
    "z4",
    "z1z3",
    "z1z4",
    "std",
    "mad",
    "maxmin",
    "top3_mean",
    "top3_std",
)


def build_features(sorted_scores: np.ndarray) -> np.ndarray:
    """sorted_scores: (n, 10) descending step-16 scores -> (n, len(FEATURES))."""
    s = np.asarray(sorted_scores, dtype=float)
    median = np.median(s, axis=1, keepdims=True)
    mad = np.median(np.abs(s - median), axis=1, keepdims=True) + EPS
    z = (s - median) / mad
    columns = {
        "g1": z[:, 0] - z[:, 1],
        "g2": z[:, 1] - z[:, 2],
        "g3": z[:, 2] - z[:, 3],
        "z1": z[:, 0],
        "z2": z[:, 1],
        "z3": z[:, 2],
        "z4": z[:, 3],
        "z1z3": z[:, 0] - z[:, 2],
        "z1z4": z[:, 0] - z[:, 3],
        "std": s.std(axis=1),
        "mad": mad[:, 0],
        "maxmin": s[:, 0] - s[:, -1],
        "top3_mean": s[:, :3].mean(axis=1),
        "top3_std": s[:, :3].std(axis=1),
    }
    return np.column_stack([columns[name] for name in FEATURES])


class IsotonicModelA:
    """Model A: one isotonic regression per K on the boundary gap g_K."""

    def __init__(self) -> None:
        self.models: list[IsotonicRegression] = []

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "IsotonicModelA":
        gap_columns = [FEATURES.index("g1"), FEATURES.index("g2"), FEATURES.index("g3")]
        self.models = []
        for k in range(3):
            model = IsotonicRegression(increasing=False, out_of_bounds="clip")
            model.fit(features[:, gap_columns[k]], targets[:, k])
            self.models.append(model)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        gap_columns = [FEATURES.index("g1"), FEATURES.index("g2"), FEATURES.index("g3")]
        predicted = np.column_stack(
            [self.models[k].predict(features[:, gap_columns[k]]) for k in range(3)]
        )
        return np.maximum(predicted, 0.0)


class BoostedModelB:
    """Model B: one gradient-boosted regressor per K on all features."""

    def __init__(self, random_state: int = 20260919) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        self.models = [
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                min_samples_leaf=50,
                random_state=random_state + k,
            )
            for k in range(3)
        ]

    def feature_importance(self) -> np.ndarray:
        return np.mean([model.feature_importances_ for model in self.models], axis=0)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "BoostedModelB":
        for k, model in enumerate(self.models):
            model.fit(features, targets[:, k])
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        predicted = np.column_stack([model.predict(features) for model in self.models])
        return np.maximum(predicted, 0.0)


def project_monotone(predicted: np.ndarray) -> np.ndarray:
    """Exact PAVA projection onto Lhat1 >= Lhat2 >= Lhat3, clipped at zero.

    All predictions are first clipped to be non-negative; PAVA pooling of
    non-negative values stays non-negative, so the result is feasible.
    """
    out = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    a, b, c = out[:, 0], out[:, 1], out[:, 2]
    m_all = (a + b + c) / 3.0
    m_left = (a + b) / 2.0
    m_right = (b + c) / 2.0
    result = out.copy()
    case_all = (a < b) & (b < c)
    case_left = (a < b) & (b >= c) & (m_left < c)
    case_right = (a >= b) & (b < c) & (a < m_right)
    left_ok = (a < b) & (b >= c) & (m_left >= c)
    right_ok = (a >= b) & (b < c) & (a >= m_right)
    result[left_ok, 0] = m_left[left_ok]
    result[left_ok, 1] = m_left[left_ok]
    result[left_ok, 2] = c[left_ok]
    result[right_ok, 0] = a[right_ok]
    result[right_ok, 1] = m_right[right_ok]
    result[right_ok, 2] = m_right[right_ok]
    pooled = case_all | case_left | case_right
    result[pooled, 0] = m_all[pooled]
    result[pooled, 1] = m_all[pooled]
    result[pooled, 2] = m_all[pooled]
    return result


def exact_budget_allocation(costs: np.ndarray) -> np.ndarray:
    """Exact DP for min sum_p costs[p, K_p] with K_p in {1,2,3} and sum K_p = 2N.

    ``costs`` has shape (n, 3); entry [p, j] is the predicted regret of K=j+1.
    Returns an integer array of chosen K values whose sum is exactly 2n.
    """
    n = costs.shape[0]
    target = 2 * n
    inf = 1e18
    max_state = 3 * n
    dp = np.full(max_state + 1, inf)
    dp[0] = 0.0
    parents = np.zeros((n, max_state + 1), dtype=np.int8)
    for p in range(n):
        new_dp = np.full(max_state + 1, inf)
        best_k = np.zeros(max_state + 1, dtype=np.int8)
        for k in (1, 2, 3):
            source = dp[: max_state + 1 - k] + costs[p, k - 1]
            target_indices = np.arange(k, max_state + 1)
            better = source < new_dp[target_indices]
            new_dp[target_indices[better]] = source[better]
            best_k[target_indices[better]] = k
        parents[p] = best_k
        dp = new_dp
    assert dp[target] < inf, "infeasible allocation"
    chosen = np.zeros(n, dtype=int)
    state = target
    for p in range(n - 1, -1, -1):
        k = int(parents[p, state])
        chosen[p] = k
        state -= k
    assert state == 0 and chosen.sum() == target
    return chosen


def allocation_from_marginal_gains(predicted: np.ndarray) -> np.ndarray:
    """Greedy precedence-aware allocation (verification reference).

    Kept only to cross-check the exact DP on small batches.  It selects the N
    cheapest upgrades, respecting that the 2->3 upgrade requires the 1->2 upgrade.
    """
    n = predicted.shape[0]
    chosen = np.ones(n, dtype=int)
    # base cost at K=1; an upgrade 1->2 has gain b12, then 2->3 has gain b23.
    b12 = predicted[:, 0] - predicted[:, 1]
    b23 = predicted[:, 1] - predicted[:, 2]
    # Each prompt contributes a two-step upgrade with gains (b12, b23).
    # Exact optimum for this convex chain is obtained by a priority queue on the
    # next available gain; ties broken by prompt index for determinism.
    import heapq

    heap = []
    for p in range(n):
        heapq.heappush(heap, (b12[p], p, 1))
    upgrades = 0
    while upgrades < n and heap:
        gain, p, stage = heapq.heappop(heap)
        if stage == 1:
            chosen[p] = 2
            upgrades += 1
            heapq.heappush(heap, (b23[p], p, 2))
        else:
            chosen[p] = 3
            upgrades += 1
    return chosen
