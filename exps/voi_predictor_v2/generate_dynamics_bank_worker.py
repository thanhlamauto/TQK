#!/usr/bin/env python3
"""Dynamics calibration bank for predictor v2 (F2/F3).

For each of the 200 independent calibration prompts, run 25 candidates to step 64
with deterministic DDIM (T=64, eta=0), scoring ImageReward at steps 12, 14, 16, 32
and the final step, and additionally storing:

* per-candidate predicted-clean latent drift ``||x0_16 - x0_14|| / ||x0_16||``;
* the step-16 ``x0_hat`` latents (fp16) for top-4 pairwise diversity;
* frozen 768-d ImageReward hidden features at step 16.

This is a calibration bank only; it never uses GenEval prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
TEXT = ROOT / "Fk-Diffusion-Steering" / "text_to_image"
for path in (TEXT, TEXT / "fkd_diffusers"):
    sys.path.insert(0, str(path))

from diffusers import DDIMScheduler  # noqa: E402
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion, latent_to_decode  # noqa: E402
from fkd_diffusers.rewards import REWARDS_DICT, do_image_reward  # noqa: E402
from ir_features import extract_ir_hidden  # noqa: E402

MODEL = "runwayml/stable-diffusion-v1-5"
STEPS = 64
ETA = 0.0
GUIDANCE = 7.5
CANDIDATES = 25
SCORE_STEPS = (12, 14, 16, 32)
DYNAMICS_STEPS = (14, 16)
FINAL_STEP = 64
SEED_BASE = 20260920


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def latent_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_prompts() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (EXP / "prompts/calibration_prompts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 200
    return rows


def build_pipeline():
    pipe = FKDStableDiffusion.from_pretrained(MODEL, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def explicit_pool(pipe, prompt_id: int):
    seeds = [SEED_BASE + prompt_id * 32 + cid for cid in range(CANDIDATES)]
    generators = [torch.Generator(device="cuda:0").manual_seed(seed) for seed in seeds]
    pool = pipe.prepare_latents(
        batch_size=CANDIDATES,
        num_channels_latents=pipe.unet.config.in_channels,
        height=pipe.unet.config.sample_size * pipe.vae_scale_factor,
        width=pipe.unet.config.sample_size * pipe.vae_scale_factor,
        dtype=pipe.unet.dtype,
        device=torch.device("cuda:0"),
        generator=generators,
        latents=None,
    )
    return pool, seeds, [latent_hash(pool[i]) for i in range(CANDIDATES)]


def generate_prompt(pipe, row: dict[str, Any], worker_index: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    prompt_id = int(row["prompt_id"])
    prompt = row["prompt"]
    pool, seeds, hashes = explicit_pool(pipe, prompt_id)
    generators = [torch.Generator(device="cuda:0").manual_seed(seed) for seed in seeds]
    scores: dict[int, list[float]] = {}
    latents: dict[int, np.ndarray] = {}
    hidden16: list[np.ndarray] = []
    requested = set(SCORE_STEPS) | {FINAL_STEP}

    def callback(_pipe, step_index, _timestep, kwargs):
        step = int(step_index) + 1
        if step not in requested:
            return {"latents": kwargs["latents"]}
        x0 = kwargs["x0_preds"]
        decoded = latent_to_decode(model=pipe, output_type="pil", latents=x0)
        values = do_image_reward(prompts=[prompt] * CANDIDATES, image_tensors=decoded)
        scores[step] = [float(value) for value in values]
        if step in DYNAMICS_STEPS:
            latents[step] = x0.detach().to(torch.float16).cpu().numpy()
        if step == 16:
            model = REWARDS_DICT["ImageReward"]
            with torch.no_grad():
                features = extract_ir_hidden(model, [prompt] * CANDIDATES, decoded)
            hidden16.append(features.detach().cpu().numpy().astype(np.float16))
            del features
        del decoded
        return {"latents": kwargs["latents"]}

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        pipe(
            prompt=[prompt] * CANDIDATES,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            eta=ETA,
            generator=generators,
            latents=pool,
            output_type="latent",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "x0_preds"],
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert set(scores) == requested and set(latents) == set(DYNAMICS_STEPS)
    assert len(hidden16) == 1

    x0_16 = latents[16].astype(np.float32).reshape(CANDIDATES, -1)
    x0_14 = latents[14].astype(np.float32).reshape(CANDIDATES, -1)
    drift = np.linalg.norm(x0_16 - x0_14, axis=1) / (np.linalg.norm(x0_16, axis=1) + 1e-6)
    candidates = []
    for cid in range(CANDIDATES):
        candidates.append({
            "candidate_id": cid,
            "seed": seeds[cid],
            "initial_latent_sha256": hashes[cid],
            **{f"IR_t{step}": scores[step][cid] for step in SCORE_STEPS},
            "IR_final": scores[FINAL_STEP][cid],
            "latent_drift_14_16": float(drift[cid]),
        })
    payload = {
        **row,
        "worker_index": worker_index,
        "model": MODEL,
        "steps": STEPS,
        "eta": ETA,
        "guidance_scale": GUIDANCE,
        "candidate_seed_base": SEED_BASE,
        "elapsed_s": elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "candidates": candidates,
    }
    arrays = {"x0_16": latents[16], "hidden16": hidden16[0]}
    return payload, arrays


def complete(prompt_id: int, worker_index: int) -> bool:
    json_path = EXP / "dynamics_raw" / f"gpu{worker_index}" / f"{prompt_id:05d}.json"
    npz_path = EXP / "dynamics_raw" / f"gpu{worker_index}" / f"{prompt_id:05d}.npz"
    return json_path.exists() and npz_path.exists()


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def warm_up(pipe, prompt: str) -> None:
    generator = [torch.Generator(device="cuda:0").manual_seed(987654)]
    with torch.inference_mode():
        pipe(prompt=[prompt], num_inference_steps=2, eta=0.0, generator=generator, output_type="latent")
    do_image_reward(images=[Image.new("RGB", (224, 224))], prompts=[prompt])
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--limit-prompts", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}")
    prompts = load_prompts()
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    owned = [row for row in prompts if int(row["prompt_id"]) % args.num_workers == args.worker_index]
    if not owned:
        print(json.dumps({"worker": args.worker_index, "completed": 0, "owned": 0}))
        return
    pipe = build_pipeline()
    warm_up(pipe, owned[0]["prompt"])
    import diffusers
    import transformers

    atomic_json(EXP / "dynamics_raw" / f"gpu{args.worker_index}" / "hardware.json", {
        "worker": args.worker_index,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "image_reward": package_version("image-reward"),
        "python": platform.python_version(),
        "owned_prompt_ids": [int(row["prompt_id"]) for row in owned],
    })
    completed = 0
    for row in owned:
        prompt_id = int(row["prompt_id"])
        if args.resume and complete(prompt_id, args.worker_index):
            completed += 1
            continue
        payload, arrays = generate_prompt(pipe, row, args.worker_index)
        out_dir = EXP / "dynamics_raw" / f"gpu{args.worker_index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_tmp = out_dir / f"{prompt_id:05d}.tmp.npz"
        np.savez(npz_tmp, **arrays)
        os.replace(npz_tmp, out_dir / f"{prompt_id:05d}.npz")
        atomic_json(out_dir / f"{prompt_id:05d}.json", payload)
        print(json.dumps({
            "phase": "dynamics_bank",
            "prompt_id": prompt_id,
            "elapsed_s": payload["elapsed_s"],
            "peak_vram_gib": payload["peak_vram_gib"],
        }), flush=True)
        completed += 1
    print(json.dumps({"worker": args.worker_index, "completed": completed, "owned": len(owned)}), flush=True)


if __name__ == "__main__":
    main()
