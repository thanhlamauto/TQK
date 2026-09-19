#!/usr/bin/env python3
"""Offline ordered 10-seed pool simulation (no new images).

For each calibration prompt, draw many ORDERED 10-candidate pools without
replacement with RNG seed 20260920, then simulate:

* Action A: 10 -> Top2@16 -> final; ``Y_A = max final over the two survivors``.
* Action B: 10 -> Top4@16 -> Top1@32 -> final; ``Y_B = final of that survivor``.
* PSP: first 8 of the ordered pool -> Top4@16 -> Top2@32 -> final;
  ``Y_PSP = max final over the two step-32 survivors``.

``V = Y_B - Y_A`` and ``Oracle = max(Y_A, Y_B)``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voi_models import FEATURES, build_features

EXP = Path(__file__).resolve().parent
CAL = EXP / "calibration"
BANK = CAL / "bank.parquet"
POOL_SEED = 20260920
POOLS_PER_PROMPT = 1000
POOL_SIZE = 10
PSP_PREFIX = 8
CANDIDATES = 25


def top_k_indices(scores: np.ndarray, candidate_ids: np.ndarray, k: int) -> np.ndarray:
    """Top-k columns per row, tie-break by candidate id ascending."""
    return np.lexsort((candidate_ids, -scores), axis=1)[:, :k]


def simulate(scores16, scores32, finals, pools):
    """Return (Y_A, Y_B, Y_PSP, A survivor ids, B winner id)."""
    s16 = scores16[pools]
    order16 = top_k_indices(s16, pools, 4)
    a_ids = order16[:, :2]
    a_candidates = np.take_along_axis(pools, a_ids, axis=1)
    y_a = finals[a_candidates].max(axis=1)
    b16 = np.take_along_axis(pools, order16, axis=1)
    s32_top4 = scores32[b16]
    b32_order = np.lexsort((b16, -s32_top4), axis=1)[:, :1]
    b_winner = np.take_along_axis(b16, b32_order, axis=1)[:, 0]
    y_b = finals[b_winner]
    psp_pool = pools[:, :PSP_PREFIX]
    psp_order16 = top_k_indices(scores16[psp_pool], psp_pool, 4)
    psp16 = np.take_along_axis(psp_pool, psp_order16, axis=1)
    psp_order32 = np.lexsort((psp16, -scores32[psp16]), axis=1)[:, :2]
    psp32 = np.take_along_axis(psp16, psp_order32, axis=1)
    y_psp = finals[psp32].max(axis=1)
    return y_a, y_b, y_psp, a_candidates, b_winner


def main() -> None:
    bank = pd.read_parquet(BANK)
    prompt_ids = sorted(bank.prompt_id.unique().tolist())
    assert len(prompt_ids) == 200
    rows = []
    for prompt_id in prompt_ids:
        group = bank[bank.prompt_id.eq(prompt_id)].sort_values("candidate_id")
        assert len(group) == CANDIDATES
        scores16 = group.IR_step16.to_numpy(dtype=float)
        scores32 = group.IR_step32.to_numpy(dtype=float)
        finals = group.IR_final.to_numpy(dtype=float)
        fold = int(group.fold.iloc[0])
        rng = np.random.default_rng([POOL_SEED, prompt_id])
        # Random ordered 10-subsets: row-argsort of uniforms, take first 10.
        pools = np.argsort(rng.random((POOLS_PER_PROMPT, CANDIDATES)), axis=1)[:, :POOL_SIZE]

        y_a, y_b, y_psp, a_candidates, b_winner = simulate(scores16, scores32, finals, pools)
        s16 = scores16[pools]
        sorted_scores = np.take_along_axis(s16, np.lexsort((pools, -s16), axis=1), axis=1)
        features = build_features(sorted_scores)
        frame = pd.DataFrame(features, columns=list(FEATURES))
        frame.insert(0, "pool_index", np.arange(POOLS_PER_PROMPT))
        frame.insert(0, "fold", fold)
        frame.insert(0, "prompt_id", prompt_id)
        frame["Y_A"] = y_a
        frame["Y_B"] = y_b
        frame["Y_PSP"] = y_psp
        frame["V"] = y_b - y_a
        frame["oracle"] = np.maximum(y_a, y_b)
        frame["action_A_winner_ids"] = ["|".join(map(str, ids)) for ids in a_candidates]
        frame["action_B_winner_id"] = b_winner
        rows.append(frame)

    dataset = pd.concat(rows, ignore_index=True)
    assert len(dataset) == len(prompt_ids) * POOLS_PER_PROMPT
    numeric = ["Y_A", "Y_B", "Y_PSP", "V", "oracle"] + list(FEATURES)
    assert np.isfinite(dataset[numeric].to_numpy()).all()
    assert (dataset.oracle + 1e-9 >= dataset[["Y_A", "Y_B"]].max(axis=1)).all()
    assert np.allclose(dataset.V, dataset.Y_B - dataset.Y_A)

    CAL.mkdir(exist_ok=True)
    out = CAL / "pool_examples.parquet"
    dataset.to_parquet(out, index=False)
    manifest = {
        "pool_seed": POOL_SEED,
        "pools_per_prompt": POOLS_PER_PROMPT,
        "pool_size": POOL_SIZE,
        "psp_prefix": PSP_PREFIX,
        "prompts": len(prompt_ids),
        "rows": int(len(dataset)),
        "features": list(FEATURES),
        "pools_choose_B_when_V_positive_fraction": float((dataset.V > 0).mean()),
        "V_mean": float(dataset.V.mean()),
        "V_prompt_level_mean": float(dataset.groupby("prompt_id").V.mean().mean()),
        "oracle_mean": float(dataset.oracle.mean()),
        "Y_A_mean": float(dataset.Y_A.mean()),
        "Y_B_mean": float(dataset.Y_B.mean()),
        "Y_PSP_mean": float(dataset.Y_PSP.mean()),
        "dataset_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (CAL / "pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "pools_ready", **manifest}, indent=2))


if __name__ == "__main__":
    main()
