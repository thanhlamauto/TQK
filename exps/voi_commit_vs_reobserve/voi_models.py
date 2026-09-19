"""Value-of-Information features, models, thresholds and NFE accounting.

Two equal-compute actions are compared at step 16:

* Action A: 10 -> 2 @16 -> final            (160 + 2*48 = 256 logical NFE)
* Action B: 10 -> 4 @16 -> 1 @32 -> final   (160 + 4*16 + 1*32 = 256 logical NFE)

PSP: 8 -> 4 @16 -> 2 @32 -> final = 256 logical NFE.  PSP uses the first 8 of the
ordered 10-candidate pool, so the initial pools are nested.

The label is ``V = Y_B - Y_A``.  The selector sees only step-16 ImageReward
scores; no step-32, final, prompt-category, HPS or GenEval information is used.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6

FEATURES = (
    "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10",
    "g1", "g2", "g3", "g4",
    "z1z3", "z1z4", "z1z5",
    "mean", "std", "mad", "maxmin",
    "top2_mean", "top4_mean", "top4_std",
)

ACTION_A_NFE = 10 * 16 + 2 * (64 - 16)
ACTION_B_NFE = 10 * 16 + 4 * (32 - 16) + 1 * (64 - 32)
PSP_NFE = 8 * 16 + 4 * (32 - 16) + 2 * (64 - 32)
assert ACTION_A_NFE == 256
assert ACTION_B_NFE == 256
assert PSP_NFE == 256


def build_features(sorted_scores: np.ndarray) -> np.ndarray:
    """sorted_scores: (n, 10) descending step-16 scores -> (n, len(FEATURES))."""
    s = np.asarray(sorted_scores, dtype=float)
    median = np.median(s, axis=1, keepdims=True)
    mad = np.median(np.abs(s - median), axis=1, keepdims=True) + EPS
    z = (s - median) / mad
    columns = {f"z{i + 1}": z[:, i] for i in range(10)}
    columns.update({f"g{i + 1}": z[:, i] - z[:, i + 1] for i in range(4)})
    columns["z1z3"] = z[:, 0] - z[:, 2]
    columns["z1z4"] = z[:, 0] - z[:, 3]
    columns["z1z5"] = z[:, 0] - z[:, 4]
    columns["mean"] = s.mean(axis=1)
    columns["std"] = s.std(axis=1)
    columns["mad"] = mad[:, 0]
    columns["maxmin"] = s[:, 0] - s[:, -1]
    columns["top2_mean"] = s[:, :2].mean(axis=1)
    columns["top4_mean"] = s[:, :4].mean(axis=1)
    columns["top4_std"] = s[:, :4].std(axis=1)
    return np.column_stack([columns[name] for name in FEATURES])


class AmbiguityBaseline:
    """Model A: isotonic regression of V on one scalar ambiguity score."""

    def __init__(self, scalar: str = "g2") -> None:
        assert scalar in ("g2", "min_g2_g4")
        self.scalar = scalar
        self.model = IsotonicRegression(increasing=True, out_of_bounds="clip")

    def _scalar(self, features: np.ndarray) -> np.ndarray:
        if self.scalar == "g2":
            return features[:, FEATURES.index("g2")]
        return np.minimum(features[:, FEATURES.index("g2")], features[:, FEATURES.index("g4")])

    def fit(self, features: np.ndarray, target: np.ndarray) -> "AmbiguityBaseline":
        self.model.fit(self._scalar(features), target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(self._scalar(features))


class RidgeModel:
    def __init__(self, alpha: float = 1.0) -> None:
        self.pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, features: np.ndarray, target: np.ndarray) -> "RidgeModel":
        self.pipeline.fit(features, target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(features)


class BoostedModel:
    def __init__(self, random_state: int = 20260920) -> None:
        self.model = HistGradientBoostingRegressor(
            max_leaf_nodes=15,
            max_iter=200,
            learning_rate=0.06,
            l2_regularization=1.0,
            min_samples_leaf=40,
            random_state=random_state,
        )

    def fit(self, features: np.ndarray, target: np.ndarray) -> "BoostedModel":
        self.model.fit(features, target)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features)


THRESHOLD_GRID = (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05)


def choose_b(v_hat: np.ndarray, delta: float) -> np.ndarray:
    return np.asarray(v_hat) > delta


def choose_threshold_on_train(
    v_hat: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 4,
) -> float:
    """Select a single threshold on training folds only, using grouped inner CV."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    unique = np.unique(groups)
    n_splits = min(n_splits, len(unique))
    splitter = GroupKFold(n_splits=n_splits)
    best_delta, best_reward = 0.0, -np.inf
    for delta in THRESHOLD_GRID:
        rewards = []
        for train_index, _ in splitter.split(v_hat, groups=groups):
            picked = choose_b(v_hat[train_index], delta)
            reward = np.where(picked, y_b[train_index], y_a[train_index]).mean()
            rewards.append(reward)
        mean_reward = float(np.mean(rewards))
        if mean_reward > best_reward:
            best_reward, best_delta = mean_reward, float(delta)
    return best_delta


def prompt_aggregate(values: np.ndarray, prompt_ids: np.ndarray) -> dict:
    """Mean within prompt first, then mean over prompts (section 18)."""
    order = np.argsort(prompt_ids, kind="stable")
    sorted_prompts = prompt_ids[order]
    sorted_values = values[order]
    unique, counts = np.unique(sorted_prompts, return_counts=True)
    sums = np.add.reduceat(sorted_values, np.concatenate([[0], np.cumsum(counts)[:-1]]))
    per_prompt = sums / counts
    return {
        "mean": float(per_prompt.mean()),
        "per_prompt": per_prompt,
        "prompt_ids": unique,
    }
