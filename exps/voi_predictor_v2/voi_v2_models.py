"""Predictor v2: weighted policy classification and temporal / dynamics features.

Objective change versus v1: instead of regressing ``V = Y_B - Y_A`` with MSE, we
train a *weighted classifier* for ``z = 1[V > 0]`` with weight ``w = |V|``
(capped at the 95th percentile of |V|).  A Bayes-optimal weighted classifier
chooses B exactly when ``E[V | X] > 0``, which aligns training with the final
ImageReward objective.

Feature families (pre-registered, exactly four):

* F0 - scalar step-16 geometry (the v1 baseline).
* F1 - F0 + temporal reward dynamics from steps 12 and 14 (no extra UNet NFE;
  only extra ImageReward calls).
* F2 - F0 + predicted-clean latent drift 14->16 (no extra verifier calls;
  requires a trajectory bank that stores latents).
* F3 - F1 + latent drift + ImageReward hidden features at step 16.

F2/F3 require a dynamics bank; F0/F1 are computed entirely from the existing
calibration bank.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6
N_CANDIDATES = 10

# F0: scalar step-16 geometry.
F0_FEATURES = (
    "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10",
    "g1", "g2", "g3", "g4",
    "z1z3", "z1z4", "z1z5",
    "mean", "std", "mad", "maxmin",
    "top2_mean", "top4_mean", "top4_std",
)

# F1 additions: temporal reward dynamics from steps 12/14/16.
F1_EXTRA = (
    "g12_1", "g12_2", "g12_3", "g12_4",
    "g14_1", "g14_2", "g14_3", "g14_4",
    "dgap_14_16_1", "dgap_14_16_2", "dgap_14_16_3", "dgap_14_16_4",
    "dgap_12_14_1", "dgap_12_14_2", "dgap_12_14_3", "dgap_12_14_4",
    "v_rank1", "v_rank2", "v_rank3", "v_rank4",
    "a_rank1", "a_rank2", "a_rank3", "a_rank4",
    "m_rescue",
    "tau_12_16", "tau_14_16",
    "churn_12_16", "churn_14_16",
    "mean12", "mean14", "std12", "std14",
)


def _z_scores(scores: np.ndarray) -> np.ndarray:
    median = np.median(scores, axis=1, keepdims=True)
    mad = np.median(np.abs(scores - median), axis=1, keepdims=True) + EPS
    return (scores - median) / mad


def build_f0(sorted_scores16: np.ndarray) -> np.ndarray:
    s = np.asarray(sorted_scores16, dtype=float)
    z = _z_scores(s)
    columns = {f"z{i + 1}": z[:, i] for i in range(10)}
    columns.update({f"g{i + 1}": z[:, i] - z[:, i + 1] for i in range(4)})
    columns["z1z3"] = z[:, 0] - z[:, 2]
    columns["z1z4"] = z[:, 0] - z[:, 3]
    columns["z1z5"] = z[:, 0] - z[:, 4]
    columns["mean"] = s.mean(axis=1)
    columns["std"] = s.std(axis=1)
    median = np.median(s, axis=1, keepdims=True)
    columns["mad"] = (np.median(np.abs(s - median), axis=1) + EPS)
    columns["maxmin"] = s[:, 0] - s[:, -1]
    columns["top2_mean"] = s[:, :2].mean(axis=1)
    columns["top4_mean"] = s[:, :4].mean(axis=1)
    columns["top4_std"] = s[:, :4].std(axis=1)
    return np.column_stack([columns[name] for name in F0_FEATURES])


def _kendall_tau(rank_a: np.ndarray, rank_b: np.ndarray) -> np.ndarray:
    """Vectorised Kendall tau (no ties) over the last axis."""
    n = rank_a.shape[1]
    total = np.zeros(rank_a.shape[0], dtype=float)
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += np.sign(rank_a[:, i] - rank_a[:, j]) * np.sign(rank_b[:, i] - rank_b[:, j])
            pairs += 1
    return total / pairs


def _rank_rows(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1])[None, :]
    return ranks


def _top4_churn(ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    match = (ids_a[:, :, None] == ids_b[:, None, :]).any(axis=2).sum(axis=1)
    return 1.0 - match / (8.0 - match)


def build_f1(
    scores12: np.ndarray,
    scores14: np.ndarray,
    scores16: np.ndarray,
    ids: np.ndarray,
) -> np.ndarray:
    """scores*: (n,10) values aligned with ``ids`` (n,10) candidate ids."""
    order16 = np.lexsort((ids, -scores16), axis=1)
    sorted16 = np.take_along_axis(scores16, order16, axis=1)
    sorted14 = np.take_along_axis(scores14, order16, axis=1)
    sorted12 = np.take_along_axis(scores12, order16, axis=1)
    ids_sorted = np.take_along_axis(ids, order16, axis=1)

    order14 = np.lexsort((ids, -scores14), axis=1)[:, :4]
    order12 = np.lexsort((ids, -scores12), axis=1)[:, :4]
    top4_14 = np.take_along_axis(ids, order14, axis=1)
    top4_12 = np.take_along_axis(ids, order12, axis=1)
    top4_16 = ids_sorted[:, :4]

    g16 = sorted16[:, :-1] - sorted16[:, 1:]
    g14 = sorted14[:, :-1] - sorted14[:, 1:]
    g12 = sorted12[:, :-1] - sorted12[:, 1:]
    v = sorted16 - sorted14
    a = v - (sorted14 - sorted12)

    columns = {f"g12_{k + 1}": g12[:, k] for k in range(4)}
    columns.update({f"g14_{k + 1}": g14[:, k] for k in range(4)})
    columns.update({f"dgap_14_16_{k + 1}": g16[:, k] - g14[:, k] for k in range(4)})
    columns.update({f"dgap_12_14_{k + 1}": g14[:, k] - g12[:, k] for k in range(4)})
    columns.update({f"v_rank{k + 1}": v[:, k] for k in range(4)})
    columns.update({f"a_rank{k + 1}": a[:, k] for k in range(4)})
    columns["m_rescue"] = v[:, 2:4].max(axis=1) - v[:, :2].min(axis=1)
    columns["tau_12_16"] = _kendall_tau(_rank_rows(scores12), _rank_rows(scores16))
    columns["tau_14_16"] = _kendall_tau(_rank_rows(scores14), _rank_rows(scores16))
    columns["churn_12_16"] = _top4_churn(top4_12, top4_16)
    columns["churn_14_16"] = _top4_churn(top4_14, top4_16)
    columns["mean12"] = scores12.mean(axis=1)
    columns["mean14"] = scores14.mean(axis=1)
    columns["std12"] = scores12.std(axis=1)
    columns["std14"] = scores14.std(axis=1)
    return np.column_stack([columns[name] for name in F1_EXTRA])


F2_FEATURES = (
    "drift_mean_top4",
    "drift_max_top4",
    "drift_std_top4",
    "drift_rank34_minus_rank12",
    "latent_div_top4",
)


def build_f2(drift: np.ndarray, x0_16: np.ndarray, pools: np.ndarray, order16: np.ndarray) -> np.ndarray:
    """drift (25,), x0_16 (25, D), pools (n, 10), order16 (n, 10) sorted by step 16."""
    drift_pool = drift[pools]
    drift_sorted = np.take_along_axis(drift_pool, order16, axis=1)
    top4 = drift_sorted[:, :4]
    x_top4 = x0_16[pools][np.arange(pools.shape[0])[:, None], order16[:, :4]]  # (n,4,D)
    x_top4 = x_top4.astype(np.float32)
    norms = np.linalg.norm(x_top4, axis=2, keepdims=True) + 1e-6
    normalized = x_top4 / norms
    gram = np.einsum("nid,njd->nij", normalized, normalized)
    off = gram[:, 0, 1] + gram[:, 0, 2] + gram[:, 0, 3] + gram[:, 1, 2] + gram[:, 1, 3] + gram[:, 2, 3]
    diversity = 1.0 - off / 6.0
    return np.column_stack([
        top4.mean(axis=1),
        top4.max(axis=1),
        top4.std(axis=1),
        drift_sorted[:, 2:4].mean(axis=1) - drift_sorted[:, :2].mean(axis=1),
        diversity,
    ])


def cap_weights(abs_v: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    cap = float(np.quantile(abs_v, quantile))
    return np.minimum(abs_v, cap)


class WeightedLogistic:
    def __init__(self) -> None:
        self.pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )

    def fit(self, features, label, weight):
        self.pipeline.fit(features, label, logisticregression__sample_weight=weight)
        return self

    def predict_proba(self, features):
        return self.pipeline.predict_proba(features)[:, 1]


class WeightedBoost:
    def __init__(self, random_state: int = 20260920) -> None:
        self.model = HistGradientBoostingClassifier(
            max_leaf_nodes=15,
            max_iter=200,
            learning_rate=0.06,
            l2_regularization=1.0,
            min_samples_leaf=40,
            random_state=random_state,
        )

    def fit(self, features, label, weight):
        self.model.fit(features, label, sample_weight=weight)
        return self

    def predict_proba(self, features):
        return self.model.predict_proba(features)[:, 1]


def threshold_grid(probabilities: np.ndarray) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, 51)
    return np.unique(np.quantile(probabilities, quantiles))


def best_threshold(probabilities, y_a, y_b) -> float:
    """Choose the threshold maximising the realised policy value on given data."""
    best_delta, best_value = 0.5, -np.inf
    for delta in threshold_grid(probabilities):
        picked = probabilities > delta
        value = float(np.where(picked, y_b, y_a).mean())
        if value > best_value:
            best_value, best_delta = value, float(delta)
    return best_delta


def policy_value(probabilities, y_a, y_b, delta) -> float:
    picked = probabilities > delta
    return float(np.where(picked, y_b, y_a).mean())


def oracle_headroom_capture(q_policy: float, q_a: float, q_b: float, q_oracle: float) -> float:
    baseline = max(q_a, q_b)
    denominator = q_oracle - baseline
    if denominator <= 0:
        return float("nan")
    return (q_policy - baseline) / denominator
