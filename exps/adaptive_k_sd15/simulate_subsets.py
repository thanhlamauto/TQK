#!/usr/bin/env python3
"""Offline 10-seed pool simulation (no new images).

For every calibration prompt, draw 1000 subsets of size 10 with the fixed RNG
seed 20260919, then compute:

* step-16 geometry features (see ``regret_models.build_features``);
* realized regrets ``L1, L2, L3`` from the nested Top-K selections;
* fixed 10->1 / 10->2 / 10->3 realized final rewards;
* fixed PSP ``8->4@16->2@32`` replay, where PSP uses the 8 lowest-candidate-id
  members of the subset (mirroring the online "candidates 0..7" pool).

Outputs a subset-level dataset for prompt-grouped cross validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from regret_models import FEATURES, build_features

EXP = Path(__file__).resolve().parent
CAL = EXP / "calibration"
BANK = CAL / "bank.parquet"
SUBSET_SEED = 20260919
SUBSETS_PER_PROMPT = 1000
POOL_SIZE = 10
CANDIDATES = 25


def psp_replay(scores16: np.ndarray, scores32: np.ndarray, finals: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    """PSP on the 8 lowest-candidate-id members of each 10-subset."""
    eight = np.sort(subsets, axis=1)[:, :8]
    s16 = scores16[eight]
    order16 = np.lexsort((eight, -s16), axis=1)[:, :4]
    ids4 = np.take_along_axis(eight, order16, axis=1)
    s32 = scores32[ids4]
    order32 = np.lexsort((ids4, -s32), axis=1)[:, :2]
    ids2 = np.take_along_axis(ids4, order32, axis=1)
    return finals[ids2].max(axis=1)


def main() -> None:
    bank = pd.read_parquet(BANK)
    prompt_ids = sorted(bank.prompt_id.unique().tolist())
    assert len(prompt_ids) == 200
    all_features = []
    all_labels = []
    all_meta = []
    subset_store = np.zeros((len(prompt_ids), SUBSETS_PER_PROMPT, POOL_SIZE), dtype=np.int16)
    for position, prompt_id in enumerate(prompt_ids):
        group = bank[bank.prompt_id.eq(prompt_id)].sort_values("candidate_id")
        assert len(group) == CANDIDATES
        scores16 = group.IR_step16.to_numpy(dtype=float)
        scores32 = group.IR_step32.to_numpy(dtype=float)
        finals = group.IR_final.to_numpy(dtype=float)
        fold = int(group.fold.iloc[0])
        rng = np.random.default_rng([SUBSET_SEED, prompt_id])
        subsets = np.argsort(rng.random((SUBSETS_PER_PROMPT, CANDIDATES)), axis=1)[:, :POOL_SIZE]
        subset_store[position] = subsets.astype(np.int16)

        subset_scores = scores16[subsets]
        order = np.lexsort((subsets, -subset_scores), axis=1)
        sorted_scores = np.take_along_axis(subset_scores, order, axis=1)
        sorted_ids = np.take_along_axis(subsets, order, axis=1)
        sorted_finals = finals[sorted_ids]
        oracle = finals[subsets].max(axis=1)
        topk = np.column_stack([
            sorted_finals[:, :1].max(axis=1),
            sorted_finals[:, :2].max(axis=1),
            sorted_finals[:, :3].max(axis=1),
        ])
        regret = oracle[:, None] - topk
        psp_final = psp_replay(scores16, scores32, finals, subsets)

        features = build_features(sorted_scores)
        all_features.append(features)
        all_labels.append(regret)
        all_meta.append(np.column_stack([
            np.full(SUBSETS_PER_PROMPT, prompt_id),
            np.full(SUBSETS_PER_PROMPT, fold),
            np.arange(SUBSETS_PER_PROMPT),
            topk,
            oracle,
            psp_final,
        ]))

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    meta = np.concatenate(all_meta, axis=0)
    frame = pd.DataFrame(features, columns=list(FEATURES))
    frame.insert(0, "subset_index", meta[:, 2].astype(int))
    frame.insert(0, "fold", meta[:, 1].astype(int))
    frame.insert(0, "prompt_id", meta[:, 0].astype(int))
    frame["L1"] = labels[:, 0]
    frame["L2"] = labels[:, 1]
    frame["L3"] = labels[:, 2]
    frame["topk1"] = meta[:, 3]
    frame["topk2"] = meta[:, 4]
    frame["topk3"] = meta[:, 5]
    frame["oracle"] = meta[:, 6]
    frame["psp_final"] = meta[:, 7]
    assert len(frame) == len(prompt_ids) * SUBSETS_PER_PROMPT
    assert np.isfinite(frame[list(FEATURES) + ["L1", "L2", "L3"]].to_numpy()).all()

    violation_12 = float((frame.L1 + 1e-12 < frame.L2).mean())
    violation_23 = float((frame.L2 + 1e-12 < frame.L3).mean())
    positive = {
        "L1_min": float(frame.L1.min()),
        "L2_min": float(frame.L2.min()),
        "L3_min": float(frame.L3.min()),
    }
    assert positive["L1_min"] >= -1e-9 and positive["L2_min"] >= -1e-9 and positive["L3_min"] >= -1e-9

    CAL.mkdir(exist_ok=True)
    dataset = CAL / "subset_dataset.parquet"
    frame.to_parquet(dataset, index=False)
    np.savez_compressed(CAL / "subset_indices.npz", subsets=subset_store, prompt_ids=np.array(prompt_ids))
    manifest = {
        "subset_seed": SUBSET_SEED,
        "subsets_per_prompt": SUBSETS_PER_PROMPT,
        "pool_size": POOL_SIZE,
        "candidates": CANDIDATES,
        "rows": int(len(frame)),
        "prompts": len(prompt_ids),
        "features": list(FEATURES),
        "monotonicity_violation_rate_L1_lt_L2": violation_12,
        "monotonicity_violation_rate_L2_lt_L3": violation_23,
        "label_minima": positive,
        "dataset_sha256": __import__("hashlib").sha256(dataset.read_bytes()).hexdigest(),
    }
    (CAL / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "subsets_ready", **manifest}, indent=2))


if __name__ == "__main__":
    main()
