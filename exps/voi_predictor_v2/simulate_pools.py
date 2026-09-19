#!/usr/bin/env python3
"""Ordered 10-seed pools with A/B/PSP replay and F0/F1 features.

Reuses the existing independent calibration bank (steps 12/14/16/32/64).  Pools
are ordered 10-subsets of the 25 candidates, RNG seed 20260920; size-10 pools do
not undersample the reward head, and PSP uses the first 8 of each ordered pool.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from voi_v2_models import F0_FEATURES, F1_EXTRA, build_f0, build_f1

EXP = Path(__file__).resolve().parent
CAL = EXP / "calibration"
BANK = CAL / "bank.parquet"
POOL_SEED = 20260920
POOLS_PER_PROMPT = 1000
POOL_SIZE = 10
PSP_PREFIX = 8
CANDIDATES = 25


def top_k(scores, ids, k):
    return np.lexsort((ids, -scores), axis=1)[:, :k]


def simulate(scores16, scores32, finals, pools):
    s16 = scores16[pools]
    order16 = top_k(s16, pools, 4)
    a_candidates = np.take_along_axis(pools, order16[:, :2], axis=1)
    y_a = finals[a_candidates].max(axis=1)
    b16 = np.take_along_axis(pools, order16, axis=1)
    b32 = np.lexsort((b16, -scores32[b16]), axis=1)[:, :1]
    b_winner = np.take_along_axis(b16, b32, axis=1)[:, 0]
    y_b = finals[b_winner]
    psp_pool = pools[:, :PSP_PREFIX]
    psp16 = np.take_along_axis(psp_pool, top_k(scores16[psp_pool], psp_pool, 4), axis=1)
    psp32 = np.take_along_axis(psp16, np.lexsort((psp16, -scores32[psp16]), axis=1)[:, :2], axis=1)
    y_psp = finals[psp32].max(axis=1)
    return y_a, y_b, y_psp


def main() -> None:
    bank = pd.read_parquet(BANK)
    prompt_ids = sorted(bank.prompt_id.unique().tolist())
    assert len(prompt_ids) == 200
    frames = []
    for prompt_id in prompt_ids:
        group = bank[bank.prompt_id.eq(prompt_id)].sort_values("candidate_id")
        assert len(group) == CANDIDATES
        s12 = group.IR_t12.to_numpy(dtype=float)
        s14 = group.IR_t14.to_numpy(dtype=float)
        s16 = group.IR_t16.to_numpy(dtype=float)
        s32 = group.IR_t32.to_numpy(dtype=float)
        finals = group.IR_final.to_numpy(dtype=float)
        fold = int(group.fold.iloc[0])
        rng = np.random.default_rng([POOL_SEED, prompt_id])
        pools = np.argsort(rng.random((POOLS_PER_PROMPT, CANDIDATES)), axis=1)[:, :POOL_SIZE]
        y_a, y_b, y_psp = simulate(s16, s32, finals, pools)

        p12 = s12[pools]
        p14 = s14[pools]
        p16 = s16[pools]
        f0 = build_f0(np.take_along_axis(p16, np.lexsort((pools, -p16), axis=1), axis=1))
        f1 = build_f1(p12, p14, p16, pools)
        frame = pd.DataFrame(
            np.column_stack([f0, f1]),
            columns=list(F0_FEATURES) + list(F1_EXTRA),
        )
        frame.insert(0, "pool_index", np.arange(POOLS_PER_PROMPT))
        frame.insert(0, "fold", fold)
        frame.insert(0, "prompt_id", prompt_id)
        frame["Y_A"] = y_a
        frame["Y_B"] = y_b
        frame["Y_PSP"] = y_psp
        frame["V"] = y_b - y_a
        frame["oracle"] = np.maximum(y_a, y_b)
        frames.append(frame)
    dataset = pd.concat(frames, ignore_index=True)
    assert len(dataset) == len(prompt_ids) * POOLS_PER_PROMPT
    numeric = list(F0_FEATURES) + list(F1_EXTRA) + ["Y_A", "Y_B", "Y_PSP", "V", "oracle"]
    assert np.isfinite(dataset[numeric].to_numpy()).all()
    out = CAL / "pool_examples_f01.parquet"
    dataset.to_parquet(out, index=False)
    manifest = {
        "pool_seed": POOL_SEED,
        "pools_per_prompt": POOLS_PER_PROMPT,
        "rows": int(len(dataset)),
        "positive_rate": float((dataset.V > 0).mean()),
        "V_mean": float(dataset.V.mean()),
        "Y_A_mean": float(dataset.Y_A.mean()),
        "Y_B_mean": float(dataset.Y_B.mean()),
        "Y_PSP_mean": float(dataset.Y_PSP.mean()),
        "oracle_mean": float(dataset.oracle.mean()),
        "F0_features": list(F0_FEATURES),
        "F1_extra": list(F1_EXTRA),
        "dataset_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (CAL / "pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "pools_ready", **manifest}, indent=2))


if __name__ == "__main__":
    main()
