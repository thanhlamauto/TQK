#!/usr/bin/env python3
"""Ordered pools with F0/F1/F2 features from the dynamics bank.

Also saves the frozen step-16 ImageReward hidden features (200 x 25 x 768) for
per-fold PCA in the F3 evaluation, and the per-pool Top-4 candidate ids.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulate_pools import simulate
from voi_v2_models import F0_FEATURES, F1_EXTRA, F2_FEATURES, build_f0, build_f1, build_f2

EXP = Path(__file__).resolve().parent
CAL = EXP / "calibration"
RAW = EXP / "dynamics_raw"
POOL_SEED = 20260920
POOLS_PER_PROMPT = 1000
POOL_SIZE = 10
CANDIDATES = 25


def main() -> None:
    prompt_rows = [
        json.loads(line)
        for line in (EXP / "prompts/calibration_prompts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    prompt_ids = [int(row["prompt_id"]) for row in prompt_rows]
    hidden = np.zeros((len(prompt_ids), CANDIDATES, 768), dtype=np.float16)
    frames = []
    for position, prompt_id in enumerate(prompt_ids):
        payload = None
        npz = None
        for worker in (0, 1):
            json_path = RAW / f"gpu{worker}" / f"{prompt_id:05d}.json"
            npz_path = RAW / f"gpu{worker}" / f"{prompt_id:05d}.npz"
            if json_path.exists() and npz_path.exists():
                payload = json.loads(json_path.read_text())
                npz = np.load(npz_path)
                break
        assert payload is not None, prompt_id
        candidates = sorted(payload["candidates"], key=lambda item: item["candidate_id"])
        assert [c["candidate_id"] for c in candidates] == list(range(CANDIDATES))
        s12 = np.array([c["IR_t12"] for c in candidates], dtype=float)
        s14 = np.array([c["IR_t14"] for c in candidates], dtype=float)
        s16 = np.array([c["IR_t16"] for c in candidates], dtype=float)
        s32 = np.array([c["IR_t32"] for c in candidates], dtype=float)
        finals = np.array([c["IR_final"] for c in candidates], dtype=float)
        drift = np.array([c["latent_drift_14_16"] for c in candidates], dtype=float)
        x0_16 = npz["x0_16"].reshape(CANDIDATES, -1)
        hidden[position] = npz["hidden16"]
        fold = int(payload["fold"])
        rng = np.random.default_rng([POOL_SEED, prompt_id])
        pools = np.argsort(rng.random((POOLS_PER_PROMPT, CANDIDATES)), axis=1)[:, :POOL_SIZE]
        order16 = np.lexsort((pools, -s16[pools]), axis=1)
        y_a, y_b, y_psp = simulate(s16, s32, finals, pools)

        p16 = s16[pools]
        f0 = build_f0(np.take_along_axis(p16, order16, axis=1))
        f1 = build_f1(s12[pools], s14[pools], p16, pools)
        f2 = build_f2(drift, x0_16, pools, order16)
        top4_ids = np.take_along_axis(pools, order16[:, :4], axis=1)
        frame = pd.DataFrame(
            np.column_stack([f0, f1, f2]),
            columns=list(F0_FEATURES) + list(F1_EXTRA) + list(F2_FEATURES),
        )
        frame.insert(0, "pool_index", np.arange(POOLS_PER_PROMPT))
        frame.insert(0, "fold", fold)
        frame.insert(0, "prompt_id", prompt_id)
        for k in range(4):
            frame[f"top4_{k}"] = top4_ids[:, k]
        frame["Y_A"] = y_a
        frame["Y_B"] = y_b
        frame["Y_PSP"] = y_psp
        frame["V"] = y_b - y_a
        frame["oracle"] = np.maximum(y_a, y_b)
        frames.append(frame)
    dataset = pd.concat(frames, ignore_index=True)
    out = CAL / "pool_examples_dynamics.parquet"
    dataset.to_parquet(out, index=False)
    hidden_out = CAL / "hidden16.npz"
    np.savez(hidden_out, hidden=hidden, prompt_ids=np.array(prompt_ids))
    manifest = {
        "pool_seed": POOL_SEED,
        "rows": int(len(dataset)),
        "F2_features": list(F2_FEATURES),
        "dataset_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "hidden_sha256": hashlib.sha256(hidden_out.read_bytes()).hexdigest(),
        "hidden_shape": list(hidden.shape),
    }
    (CAL / "pool_manifest_dynamics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "dynamics_pools_ready", **manifest}, indent=2))


if __name__ == "__main__":
    main()
