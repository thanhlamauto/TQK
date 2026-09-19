#!/usr/bin/env python3
"""Build the Adaptive-K calibration bank from the independent ImageReward corpus.

The Adaptive-K protocol only needs, per trajectory, ImageReward at step 16, at
step 32 (so PSP can be replayed) and at step 64.  Those exact quantities already
exist in the independent `dense_calibration_200` bank, which used the same 200
non-GenEval ImageReward prompts.  This script extracts them into the compact
`calibration/bank.parquet` required by the Adaptive-K protocol, and validates
shape and finiteness.

Source trajectories are deterministic DDIM, T=64, eta=0, batch 25, candidate seed
`20260920 + prompt_id*32 + candidate_id`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
PROMPTS = EXP / "prompts/calibration_prompts.jsonl"
DEFAULT_SOURCE = ROOT / "exps/dense_calibration_200"
EXPECTED_PROMPTS = 200
EXPECTED_SEEDS = 25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompts() -> pd.DataFrame:
    rows = [json.loads(line) for line in PROMPTS.read_text().splitlines() if line.strip()]
    frame = pd.DataFrame(rows).sort_values("prompt_id").reset_index(drop=True)
    assert len(frame) == EXPECTED_PROMPTS
    assert frame.prompt_id.tolist() == list(range(EXPECTED_PROMPTS))
    return frame


def build(source: Path) -> pd.DataFrame:
    prompts = load_prompts()
    records = []
    for row in prompts.itertuples():
        prompt_id = int(row.prompt_id)
        payload = None
        for worker in (0, 1):
            path = source / "bank_raw" / f"gpu{worker}" / f"{prompt_id:05d}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                break
        assert payload is not None, f"missing bank trajectory for prompt {prompt_id}"
        candidates = sorted(payload["candidates"], key=lambda item: item["candidate_id"])
        assert [int(c["candidate_id"]) for c in candidates] == list(range(EXPECTED_SEEDS))
        for candidate in candidates:
            records.append({
                "prompt_id": prompt_id,
                "prompt": row.prompt,
                "fold": int(row.fold),
                "candidate_id": int(candidate["candidate_id"]),
                "seed": int(candidate["seed"]),
                "IR_step16": float(candidate["IR_t16"]),
                "IR_step32": float(candidate["IR_t32"]),
                "IR_final": float(candidate["IR_final"]),
            })
    frame = pd.DataFrame(records)
    assert len(frame) == EXPECTED_PROMPTS * EXPECTED_SEEDS, len(frame)
    assert not frame[["IR_step16", "IR_step32", "IR_final"]].isna().any().any()
    assert frame.groupby("prompt_id").size().eq(EXPECTED_SEEDS).all()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dense-bank", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    frame = build(args.source_dense_bank)
    out_dir = EXP / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "bank.parquet"
    frame.to_parquet(parquet, index=False)
    frame.to_csv(out_dir / "bank.csv", index=False)
    manifest = {
        "source_dense_bank": str(args.source_dense_bank),
        "source_prompt_manifest_sha256": sha256(EXP / "prompts/prompt_manifest.json"),
        "prompt_count": int(frame.prompt_id.nunique()),
        "seeds_per_prompt": EXPECTED_SEEDS,
        "trajectory_count": int(len(frame)),
        "columns": list(frame.columns),
        "candidate_seed_base": 20260920,
        "model": "runwayml/stable-diffusion-v1-5",
        "scheduler": "DDIM",
        "steps": 64,
        "eta": 0.0,
        "guidance_scale": 7.5,
        "bank_parquet_sha256": sha256(parquet),
    }
    (out_dir / "bank_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "bank_ready", **manifest}, indent=2))


if __name__ == "__main__":
    main()
