#!/usr/bin/env python3
"""Collect the dense 200x25 calibration trajectory bank.

Pre-registration (README section 2):

    X[p,i,t] = ImageReward(pred_original_sample at t), t=10..28
    Y[p,i]   = ImageReward(final sample at step 64)

Step 32 is additionally stored so the fixed PSP comparator can be replayed on the
same bank; it is not part of the single-stage search space.
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

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
TEXT = ROOT / "Fk-Diffusion-Steering" / "text_to_image"
for path in (TEXT, TEXT / "fkd_diffusers"):
    sys.path.insert(0, str(path))

from diffusers import DDIMScheduler  # noqa: E402
from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion, latent_to_decode  # noqa: E402
from fkd_diffusers.rewards import do_image_reward  # noqa: E402
from policies import BANK_CHECKPOINTS, FINAL_STEP, STEPS  # noqa: E402

MODEL = "runwayml/stable-diffusion-v1-5"
ETA = 0.0
GUIDANCE = 7.5
CANDIDATES = 25
CAL_SEED_BASE = 20260920


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
    assert len({int(row["prompt_id"]) for row in rows}) == 200
    assert {int(row["prompt_id"]) for row in rows} == set(range(200))
    return rows


def build_pipeline():
    pipe = FKDStableDiffusion.from_pretrained(MODEL, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def explicit_pool(pipe, prompt_id: int) -> tuple[torch.Tensor, list[int], list[str]]:
    seeds = [CAL_SEED_BASE + prompt_id * 32 + candidate_id for candidate_id in range(CANDIDATES)]
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
    return pool, seeds, [latent_hash(pool[index]) for index in range(CANDIDATES)]


def score_predicted_clean(pipe, prompt: str, x0_preds: torch.Tensor) -> list[float]:
    decoded = latent_to_decode(model=pipe, output_type="pil", latents=x0_preds)
    values = do_image_reward(
        prompts=[prompt] * int(decoded.shape[0]), image_tensors=decoded
    )
    scores = [float(value) for value in values]
    del decoded
    return scores


def generate_prompt(pipe, row: dict[str, Any], worker_index: int) -> dict[str, Any]:
    prompt_id = int(row["prompt_id"])
    prompt = row["prompt"]
    pool, seeds, hashes = explicit_pool(pipe, prompt_id)
    generators = [torch.Generator(device="cuda:0").manual_seed(seed) for seed in seeds]
    score_by_step: dict[int, list[float]] = {}
    score_elapsed_by_step: dict[int, float] = {}
    requested = set(BANK_CHECKPOINTS) | {FINAL_STEP}

    def callback(_pipe, step_index, _timestep, kwargs):
        step = int(step_index) + 1
        if step not in requested:
            return {"latents": kwargs["latents"]}
        torch.cuda.synchronize()
        started = time.perf_counter()
        scores = score_predicted_clean(pipe, prompt, kwargs["x0_preds"])
        torch.cuda.synchronize()
        assert len(scores) == CANDIDATES
        score_by_step[step] = scores
        score_elapsed_by_step[step] = time.perf_counter() - started
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
    assert set(score_by_step) == requested
    candidates = []
    for candidate_id in range(CANDIDATES):
        candidates.append(
            {
                "candidate_id": candidate_id,
                "seed": seeds[candidate_id],
                "initial_latent_sha256": hashes[candidate_id],
                **{
                    f"IR_t{step}": score_by_step[step][candidate_id]
                    for step in BANK_CHECKPOINTS
                },
                "IR_final": score_by_step[FINAL_STEP][candidate_id],
            }
        )
    return {
        **row,
        "worker_index": worker_index,
        "model": MODEL,
        "dtype": "torch.float16",
        "scheduler": type(pipe.scheduler).__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "steps": STEPS,
        "eta": ETA,
        "guidance_scale": GUIDANCE,
        "candidate_count": CANDIDATES,
        "candidate_seed_base": CAL_SEED_BASE,
        "candidate_seed_formula": f"{CAL_SEED_BASE} + prompt_id*32 + candidate_id",
        "elapsed_s": elapsed,
        "checkpoint_decode_reward_s": sum(score_elapsed_by_step.values()),
        "score_elapsed_by_step": score_elapsed_by_step,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "candidates": candidates,
    }


def complete(prompt_id: int, worker_index: int) -> bool:
    path = EXP / "bank_raw" / f"gpu{worker_index}" / f"{prompt_id:05d}.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        requested = {f"IR_t{step}" for step in BANK_CHECKPOINTS} | {"IR_final"}
        return (
            int(payload["prompt_id"]) == prompt_id
            and len(payload["candidates"]) == CANDIDATES
            and all(requested <= set(candidate) for candidate in payload["candidates"])
        )
    except Exception:
        return False


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def warm_up(pipe, prompt: str) -> None:
    generator = [torch.Generator(device="cuda:0").manual_seed(987654)]
    with torch.inference_mode():
        pipe(
            prompt=[prompt],
            num_inference_steps=2,
            eta=0.0,
            generator=generator,
            output_type="latent",
        )
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

    atomic_json(
        EXP / "bank_raw" / f"gpu{args.worker_index}" / "hardware.json",
        {
            "worker": args.worker_index,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "image_reward": package_version("image-reward"),
            "python": platform.python_version(),
            "owned_prompt_ids": [int(row["prompt_id"]) for row in owned],
        },
    )
    completed = 0
    for row in owned:
        prompt_id = int(row["prompt_id"])
        if args.resume and complete(prompt_id, args.worker_index):
            completed += 1
            continue
        result = generate_prompt(pipe, row, args.worker_index)
        atomic_json(EXP / "bank_raw" / f"gpu{args.worker_index}" / f"{prompt_id:05d}.json", result)
        print(
            json.dumps(
                {
                    "phase": "dense_calibration_bank",
                    "prompt_id": prompt_id,
                    "elapsed_s": result["elapsed_s"],
                    "scoring_s": result["checkpoint_decode_reward_s"],
                    "peak_vram_gib": result["peak_vram_gib"],
                }
            ),
            flush=True,
        )
        completed += 1
    print(json.dumps({"worker": args.worker_index, "completed": completed, "owned": len(owned)}), flush=True)


if __name__ == "__main__":
    main()
