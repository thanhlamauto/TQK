#!/usr/bin/env python3
"""Compact calibration bank for predictor v2.

Extracts ``IR_t12, IR_t14, IR_t16, IR_t32, IR_final`` per candidate from the
existing independent dense_calibration_200 bank (200 non-GenEval ImageReward
prompts, 25 seeds/prompt).  No trajectories are regenerated.
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
STEPS = (12, 14, 16, 32)
EXPECTED_PROMPTS = 200
EXPECTED_SEEDS = 25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dense-bank", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    prompts = pd.DataFrame([
        json.loads(line) for line in PROMPTS.read_text().splitlines() if line.strip()
    ]).sort_values("prompt_id").reset_index(drop=True)
    assert len(prompts) == EXPECTED_PROMPTS
    records = []
    for row in prompts.itertuples():
        prompt_id = int(row.prompt_id)
        payload = None
        for worker in (0, 1):
            path = args.source_dense_bank / "bank_raw" / f"gpu{worker}" / f"{prompt_id:05d}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                break
        assert payload is not None, prompt_id
        candidates = sorted(payload["candidates"], key=lambda item: item["candidate_id"])
        assert [int(c["candidate_id"]) for c in candidates] == list(range(EXPECTED_SEEDS))
        for candidate in candidates:
            record = {
                "prompt_id": prompt_id,
                "prompt": row.prompt,
                "fold": int(row.fold),
                "candidate_id": int(candidate["candidate_id"]),
                "seed": int(candidate["seed"]),
                "IR_final": float(candidate["IR_final"]),
            }
            for step in STEPS:
                record[f"IR_t{step}"] = float(candidate[f"IR_t{step}"])
            records.append(record)
    frame = pd.DataFrame(records)
    assert len(frame) == EXPECTED_PROMPTS * EXPECTED_SEEDS
    columns = [f"IR_t{step}" for step in STEPS] + ["IR_final"]
    assert not frame[columns].isna().any().any()
    out_dir = EXP / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "bank.parquet"
    frame.to_parquet(parquet, index=False)
    manifest = {
        "source_dense_bank": str(args.source_dense_bank),
        "prompt_count": int(frame.prompt_id.nunique()),
        "seeds_per_prompt": EXPECTED_SEEDS,
        "trajectory_count": int(len(frame)),
        "steps": list(STEPS),
        "candidate_seed_base": 20260920,
        "bank_parquet_sha256": sha256(parquet),
    }
    (out_dir / "bank_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "bank_ready", **manifest}, indent=2))


if __name__ == "__main__":
    main()
