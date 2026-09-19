#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXP = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-prompts", type=int, required=True)
    args = parser.parse_args()
    protocol = json.loads((EXP / "protocol_manifest.json").read_text())
    frozen = json.loads((EXP / "FROZEN_SCHEDULE.json").read_text())
    base_seeds = [int(value) for value in protocol["fresh_seed_bases"]]
    expected = {"psp": 256, "ours": int(frozen["logical_compute"])}
    rows = []
    for path in sorted((EXP / "metadata").glob("gpu*/*_rep*_*.json")):
        row = json.loads(path.read_text())
        rows.append(row)
        method = row["method"]
        assert row["logical_unet_evals"] == expected[method]
        assert row["base_seed"] == base_seeds[int(row["repetition"])]
        trace = row["trace"]
        if method == "psp":
            assert [(x["step"], x["batch_before"], x["batch_after"]) for x in trace] == [
                (16, 8, 4), (32, 4, 2), (64, 2, 1)
            ]
            assert row["batched_verifier_calls"] == 3 and row["verifier_candidate_scores"] == 14
        else:
            assert [(x["step"], x["batch_before"], x["batch_after"]) for x in trace] == [
                (int(frozen["checkpoint"]), int(frozen["M"]), int(frozen["K"])),
                (64, int(frozen["K"]), 1),
            ]
            assert row["batched_verifier_calls"] == 2
            assert row["verifier_candidate_scores"] == int(frozen["M"] + frozen["K"])
    expected_rows = args.expected_prompts * len(base_seeds) * 2
    assert len(rows) == expected_rows, (len(rows), expected_rows)
    for prompt_id in range(args.expected_prompts):
        for repetition in range(len(base_seeds)):
            pair = [
                row
                for row in rows
                if int(row["prompt_id"]) == prompt_id and int(row["repetition"]) == repetition
            ]
            assert {row["method"] for row in pair} == {"psp", "ours"}, (prompt_id, repetition)
            psp = next(row for row in pair if row["method"] == "psp")
            ours = next(row for row in pair if row["method"] == "ours")
            overlap = min(8, int(frozen["M"]))
            assert psp["candidate_seeds"][:overlap] == ours["candidate_seeds"][:overlap]
            assert psp["initial_latent_hashes"][:overlap] == ours["initial_latent_hashes"][:overlap]
            assert psp["worker_index"] == ours["worker_index"]
    print(json.dumps({
        "status": "PASS",
        "prompts": args.expected_prompts,
        "rows": len(rows),
        "repetitions": len(base_seeds),
        "fresh_seed_bases": base_seeds,
        "frozen_schedule": frozen["schedule_notation"],
        "ours_evals": frozen["logical_compute"],
    }, indent=2))


if __name__ == "__main__":
    main()
