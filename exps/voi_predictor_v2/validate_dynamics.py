#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXP = Path(__file__).resolve().parent
STEPS = (12, 14, 16, 32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-prompts", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(
        path
        for path in (EXP / "dynamics_raw").glob("gpu*/*.json")
        if path.name != "hardware.json"
    )
    assert len(paths) == args.expected_prompts, (len(paths), args.expected_prompts)
    seen = set()
    peak = 0.0
    for path in paths:
        payload = json.loads(path.read_text())
        prompt_id = int(payload["prompt_id"])
        assert prompt_id not in seen
        seen.add(prompt_id)
        assert len(payload["candidates"]) == 25
        for candidate in payload["candidates"]:
            for step in STEPS:
                assert math.isfinite(float(candidate[f"IR_t{step}"]))
            assert math.isfinite(float(candidate["IR_final"]))
            assert math.isfinite(float(candidate["latent_drift_14_16"]))
        npz = path.with_suffix(".npz")
        assert npz.exists()
        peak = max(peak, float(payload["peak_vram_gib"]))
    assert seen == set(range(args.expected_prompts))
    print(json.dumps({"status": "PASS", "prompts": len(paths), "peak_vram_gib": peak}, indent=2))


if __name__ == "__main__":
    main()
