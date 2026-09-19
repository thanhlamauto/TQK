#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
import types
from pathlib import Path

import torch
from PIL import Image

EXP = Path(__file__).resolve().parent
METHODS = ("psp", "ours")


def install_turtle_shim() -> None:
    if "turtle" not in sys.modules:
        shim = types.ModuleType("turtle")
        shim.forward = lambda *_args, **_kwargs: None
        sys.modules["turtle"] = shim


def write_rows(path: Path, scores: dict[tuple[str, int, int], float]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "repetition", "prompt_id", "hps"])
        writer.writeheader()
        for (method, repetition, prompt_id), score in sorted(scores.items()):
            writer.writerow({
                "method": method,
                "repetition": repetition,
                "prompt_id": prompt_id,
                "hps": score,
            })
    temporary.replace(path)


def main() -> None:
    protocol = json.loads((EXP / "protocol_manifest.json").read_text())
    repetitions = int(protocol["repetitions"])
    prompts = {
        int(row["prompt_id"]): row["prompt"]
        for row in (
            json.loads(line)
            for line in (EXP / "prompts_geneval_all_553.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    total = len(prompts) * repetitions * len(METHODS)
    output = EXP / "metrics/hps.csv"
    output.parent.mkdir(exist_ok=True)
    existing: dict[tuple[str, int, int], float] = {}
    if output.exists():
        with output.open() as handle:
            existing = {
                (row["method"], int(row["repetition"]), int(row["prompt_id"])): float(row["hps"])
                for row in csv.DictReader(handle)
            }
    missing = [
        (method, repetition, prompt_id)
        for method in METHODS
        for repetition in range(repetitions)
        for prompt_id in sorted(prompts)
        if (method, repetition, prompt_id) not in existing
    ]
    if not missing:
        assert len(existing) == total
        print(output)
        return
    install_turtle_shim()
    import huggingface_hub
    import hpsv2.img_score as hps_img  # type: ignore
    from hpsv2.src.open_clip import get_tokenizer  # type: ignore
    from hpsv2.utils import hps_version_map  # type: ignore

    hps_img.initialize_model()
    model = hps_img.model_dict["model"]
    preprocess = hps_img.model_dict["preprocess_val"]
    device = hps_img.device
    checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    tokenizer = get_tokenizer("ViT-H-14")
    del checkpoint
    batch_size = int(os.environ.get("HPS_BATCH_SIZE", "16"))
    for start in range(0, len(missing), batch_size):
        chunk = missing[start:start + batch_size]
        paths = [
            EXP / "outputs" / method / f"rep{repetition}" / f"{prompt_id:05d}.png"
            for method, repetition, prompt_id in chunk
        ]
        images = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in paths]).to(device)
        text = tokenizer([prompts[prompt_id] for _method, _repetition, prompt_id in chunk]).to(device)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            outputs = model(images, text)
            scores = (outputs["image_features"] * outputs["text_features"]).sum(dim=-1)
        for key, score in zip(chunk, scores.detach().float().cpu().tolist()):
            existing[key] = float(score)
        write_rows(output, existing)
        print(json.dumps({"hps_complete": len(existing), "hps_total": total}), flush=True)
    assert len(existing) == total
    print(output)


if __name__ == "__main__":
    main()
