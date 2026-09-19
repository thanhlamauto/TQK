"""Pre-registered policy space for the dense single-stage calibration.

The search space is fixed by the TQK README "Pre-registered next experiment":

* Stable Diffusion 1.5, deterministic DDIM, 64 steps, eta=0, fp16, guidance 7.5.
* Fixed budget ``M*t + K*(64-t) <= 256``.
* ``M`` is derived, never tuned independently::

      M(t, K) = floor((256 - K*(64 - t)) / t)

* Search exactly ``t in {10, 11, ..., 28}`` and ``K in {1, 2, 3}``: 57 policies.

The bank additionally stores ImageReward at step 32 so that the fixed multi-stage
PSP comparator ``8->4@16->2@32->1@64`` can be replayed on the same bank and the
same subset draws.  Step 32 is *not* part of the single-stage search space.
"""

from __future__ import annotations

from typing import NamedTuple

STEPS = 64
BUDGET = 256
SEARCH_CHECKPOINTS = tuple(range(10, 29))  # 10..28 inclusive
SURVIVORS = (1, 2, 3)
PSP_REPLAY_CHECKPOINTS = (16, 32)  # fixed comparator only
#: Checkpoints persisted by the trajectory bank.  Superset of the search grid so
#: the fixed PSP comparator can be replayed without a second generation pass.
BANK_CHECKPOINTS = tuple(sorted(set(SEARCH_CHECKPOINTS) | set(PSP_REPLAY_CHECKPOINTS)))
FINAL_STEP = 64


class Policy(NamedTuple):
    checkpoint: int
    K: int
    M: int

    @property
    def logical_compute(self) -> int:
        return self.M * self.checkpoint + self.K * (STEPS - self.checkpoint)

    @property
    def unused_compute(self) -> int:
        return BUDGET - self.logical_compute

    @property
    def q(self) -> float:
        return self.checkpoint / STEPS

    @property
    def ratio(self) -> float:
        return self.K / self.M

    @property
    def notation(self) -> str:
        return f"{self.M}\u2192{self.K}@{self.checkpoint}"


def initial_pool(t: int, k: int) -> int:
    return (BUDGET - k * (STEPS - t)) // t


def valid_policies() -> list[Policy]:
    rows: list[Policy] = []
    for t in SEARCH_CHECKPOINTS:
        for k in SURVIVORS:
            m = initial_pool(t, k)
            if m <= k or m > 25:
                continue
            policy = Policy(t, k, m)
            if policy.logical_compute > BUDGET:
                continue
            rows.append(policy)
    return rows


def as_dict(policy: Policy) -> dict[str, int | str]:
    return {
        "checkpoint": policy.checkpoint,
        "K": policy.K,
        "M": policy.M,
        "logical_compute": policy.logical_compute,
        "unused_compute": policy.unused_compute,
        "verifier_calls": 2,
        "verifier_candidate_scores": policy.M + policy.K,
        "notation": policy.notation,
    }


PSP = {"initial": 8, "checkpoints": ((16, 4), (32, 2), (64, 1))}
PSP_LOGICAL_COMPUTE = 8 * 16 + 4 * (32 - 16) + 2 * (64 - 32)
PSP_NOTATION = "8\u21924@16\u21922@32"
