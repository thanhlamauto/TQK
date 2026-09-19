#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

EXP = Path(__file__).resolve().parent
protocol = json.loads((EXP / "protocol_manifest.json").read_text())
repetitions = int(protocol["repetitions"])
rows = [
    json.loads(line)
    for line in (EXP / "prompts_geneval_all_553.jsonl").read_text().splitlines()
    if line.strip()
]
for method in ("psp", "ours"):
    for repetition in range(repetitions):
        name = f"{method}_rep{repetition}"
        root = EXP / "geneval_inputs" / name
        for row in rows:
            prompt_id = int(row["prompt_id"])
            folder = root / f"{prompt_id:05d}"
            samples = folder / "samples"
            samples.mkdir(parents=True, exist_ok=True)
            (folder / "metadata.jsonl").write_text(
                json.dumps({key: value for key, value in row.items() if key != "prompt_id"}) + "\n"
            )
            target = EXP / "outputs" / method / f"rep{repetition}" / f"{prompt_id:05d}.png"
            link = samples / "00000.png"
            if link.exists() or link.is_symlink():
                link.unlink()
            os.symlink(target.resolve(), link)
print("geneval inputs exported")
