#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from policies import BANK_CHECKPOINTS

EXP = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-prompts", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(
        path
        for path in (EXP / "bank_raw").glob("gpu*/*.json")
        if path.name != "hardware.json"
    )
    assert len(paths) == args.expected_prompts, (len(paths), args.expected_prompts)
    seen = set()
    peak = 0.0
    elapsed = 0.0
    scoring = 0.0
    required = [f"IR_t{step}" for step in BANK_CHECKPOINTS] + ["IR_final"]
    for path in paths:
        payload = json.loads(path.read_text())
        prompt_id = int(payload["prompt_id"])
        assert prompt_id not in seen
        seen.add(prompt_id)
        assert payload["model"] == "runwayml/stable-diffusion-v1-5"
        assert payload["steps"] == 64 and payload["eta"] == 0.0
        assert payload["candidate_seed_base"] == 20260920
        assert len(payload["candidates"]) == 25
        assert [row["candidate_id"] for row in payload["candidates"]] == list(range(25))
        for candidate in payload["candidates"]:
            assert all(math.isfinite(float(candidate[column])) for column in required)
        peak = max(peak, float(payload["peak_vram_gib"]))
        elapsed += float(payload["elapsed_s"])
        scoring += float(payload["checkpoint_decode_reward_s"])
    assert seen == set(range(args.expected_prompts))
    print(
        json.dumps(
            {
                "status": "PASS",
                "prompts": len(paths),
                "candidates_per_prompt": 25,
                "checkpoint_scores_per_candidate": len(BANK_CHECKPOINTS),
                "peak_vram_gib": peak,
                "mean_elapsed_s": elapsed / len(paths),
                "mean_checkpoint_decode_reward_s": scoring / len(paths),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
