#!/usr/bin/env python3
"""Checksum-lock the confirmatory protocol before any 553-prompt generation.

README section 7: compare the frozen schedule with unchanged PSP on all 553
official GenEval prompts using three fresh, pre-recorded candidate-pool base
seeds disjoint from calibration (20260920) and every previous validation run
(20260919).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
CAL = ROOT / "exps/dense_calibration_200"
SOURCE_PROMPTS = ROOT / "exps/direct_sd15_psp_vs_10to2_553/prompts_geneval_all_553.jsonl"
SOURCE_FROZEN = CAL / "FROZEN_SCHEDULE.json"
FRESH_SEED_BASES = (20260921, 20260922, 20260923)
REPETITIONS = len(FRESH_SEED_BASES)
DISALLOWED_SEED_BASES = (0, 20260919, 20260920)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_exact(source: Path, target: Path) -> None:
    if target.exists() and sha256(target) != sha256(source):
        raise RuntimeError(f"refusing to overwrite mismatched protocol artifact: {target}")
    if not target.exists():
        shutil.copy2(source, target)
    assert sha256(target) == sha256(source)


def main() -> None:
    if not SOURCE_FROZEN.exists():
        raise SystemExit("freeze gate has not produced a FROZEN_SCHEDULE.json; stop")
    prompts = [
        json.loads(line)
        for line in SOURCE_PROMPTS.read_text().splitlines()
        if line.strip()
    ]
    assert len(prompts) == 553
    assert [int(row["prompt_id"]) for row in prompts] == list(range(553))
    frozen = json.loads(SOURCE_FROZEN.read_text())
    compute = frozen["M"] * frozen["checkpoint"] + frozen["K"] * (64 - frozen["checkpoint"])
    assert compute == frozen["logical_compute"] <= 256
    assert frozen["candidate_seed_base"] == 20260920
    for base in FRESH_SEED_BASES:
        assert base not in DISALLOWED_SEED_BASES
    assert len(set(FRESH_SEED_BASES)) == REPETITIONS

    copy_exact(SOURCE_PROMPTS, EXP / "prompts_geneval_all_553.jsonl")
    copy_exact(SOURCE_FROZEN, EXP / "FROZEN_SCHEDULE.json")
    manifest = {
        "prompt_count": 553,
        "repetitions": REPETITIONS,
        "fresh_seed_bases": list(FRESH_SEED_BASES),
        "candidate_seed_formula": "base + prompt_id*32 + candidate_id",
        "disallowed_reused_seed_bases": list(DISALLOWED_SEED_BASES),
        "prompts_sha256": sha256(EXP / "prompts_geneval_all_553.jsonl"),
        "source_prompts_sha256": sha256(SOURCE_PROMPTS),
        "frozen_schedule_sha256": sha256(EXP / "FROZEN_SCHEDULE.json"),
        "source_frozen_schedule_sha256": sha256(SOURCE_FROZEN),
        "psp_schedule": "8->4@16->2@32->1@64",
        "frozen_schedule_notation": frozen["schedule_notation"],
        "schedule": frozen,
    }
    (EXP / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "protocol_locked", **{k: v for k, v in manifest.items() if k != "schedule"}}, indent=2))


if __name__ == "__main__":
    main()
