#!/usr/bin/env python3
"""Select the fixed 200-prompt independent calibration bank.

Pre-registration (README section 2): "Select exactly 200 prompts once from the
repository ImageReward corpus using a new fixed selection seed. Record the prompt
IDs and verify zero exact overlap with all 553 GenEval prompts."

The five-fold split used by the out-of-fold selector is also frozen here so that
it cannot be revised after observing rewards.  The selection seed (20260920) and
the fold seed (20260920) are new relative to the completed calibration
(20260918) and the completed confirmatory validation (base seed 20260919).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SOURCE = ROOT / "Fk-Diffusion-Steering/text_to_image/prompt_files/test_ir.json"
GENEVAL = ROOT / "Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl"
PRIOR_CALIBRATION = (
    ROOT / "exps/single_stage_calibration/prompts/calibration_prompts.jsonl"
)
SELECTION_SEED = 20260920
FOLD_SEED = 20260920
PROMPT_COUNT = 200
NUM_FOLDS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_source() -> list[dict]:
    source_rows = json.loads(SOURCE.read_text())
    unique: list[dict] = []
    seen: set[str] = set()
    for source_index, row in enumerate(source_rows):
        prompt = row["prompt"].strip()
        key = prompt.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {"source_index": source_index, "source_id": row["id"], "prompt": prompt}
        )
    return unique


def prior_calibration_prompts() -> set[str]:
    if not PRIOR_CALIBRATION.exists():
        return set()
    keys = set()
    for line in PRIOR_CALIBRATION.read_text().splitlines():
        if not line.strip():
            continue
        keys.add(json.loads(line)["prompt"].strip().casefold())
    return keys


def expected() -> tuple[list[dict], dict]:
    unique = unique_source()
    geneval_prompts = {
        json.loads(line)["prompt"].strip().casefold()
        for line in GENEVAL.read_text().splitlines()
        if line.strip()
    }
    assert not ({row["prompt"].casefold() for row in unique} & geneval_prompts)
    indices = list(range(len(unique)))
    random.Random(SELECTION_SEED).shuffle(indices)
    selected = [unique[index] for index in indices[:PROMPT_COUNT]]

    fold_order = list(range(PROMPT_COUNT))
    random.Random(FOLD_SEED).shuffle(fold_order)
    fold_of = [0] * PROMPT_COUNT
    for rank, position in enumerate(fold_order):
        fold_of[position] = rank % NUM_FOLDS

    prior = prior_calibration_prompts()
    prior_overlap = sum(row["prompt"].casefold() in prior for row in selected)

    rows = []
    for prompt_id, row in enumerate(selected):
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": row["prompt"],
                "source_corpus": "ImageReward test_ir.json",
                "source_index": row["source_index"],
                "source_id": row["source_id"],
                "fold": fold_of[prompt_id],
            }
        )
    manifest = {
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "geneval_path": str(GENEVAL.relative_to(ROOT)),
        "geneval_sha256": sha256(GENEVAL),
        "selection_seed": SELECTION_SEED,
        "fold_seed": FOLD_SEED,
        "prompt_count": PROMPT_COUNT,
        "num_folds": NUM_FOLDS,
        "fold_sizes": [
            sum(1 for row in rows if row["fold"] == fold) for fold in range(NUM_FOLDS)
        ],
        "exact_geneval_prompt_overlap": 0,
        "exact_overlap_with_previous_120_prompt_calibration": prior_overlap,
        "prompt_ids_by_fold": {
            str(fold): [
                row["prompt_id"] for row in rows if row["fold"] == fold
            ]
            for fold in range(NUM_FOLDS)
        },
    }
    return rows, manifest


def render_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    rows, manifest = expected()
    prompt_path = EXP / "prompts/calibration_prompts.jsonl"
    manifest_path = EXP / "prompts/prompt_manifest.json"
    expected_prompts = render_jsonl(rows)
    expected_manifest = json.dumps(manifest, indent=2) + "\n"
    if args.verify_existing:
        assert prompt_path.read_text() == expected_prompts
        assert manifest_path.read_text() == expected_manifest
        print(json.dumps({"status": "verified", **manifest}))
        return
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(expected_prompts)
    manifest_path.write_text(expected_manifest)
    print(json.dumps({"status": "written", **manifest}))


if __name__ == "__main__":
    main()
