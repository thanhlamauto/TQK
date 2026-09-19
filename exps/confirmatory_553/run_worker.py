#!/usr/bin/env python3
"""Confirmatory 553-prompt, three-fresh-seed comparison.

For every prompt and every pre-recorded base seed the frozen single-stage
schedule and unchanged PSP are run from the *same* candidate pool so the
per-prompt difference is paired.
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

MODEL = "runwayml/stable-diffusion-v1-5"
STEPS = 64
ETA = 0.0
GUIDANCE = 7.5


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def latent_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_protocol() -> dict[str, Any]:
    return json.loads((EXP / "protocol_manifest.json").read_text())


def load_frozen() -> dict[str, Any]:
    frozen = json.loads((EXP / "FROZEN_SCHEDULE.json").read_text())
    compute = frozen["M"] * frozen["checkpoint"] + frozen["K"] * (STEPS - frozen["checkpoint"])
    assert compute == frozen["logical_compute"] <= 256
    return frozen


def schedules(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "psp": {"initial": 8, "checkpoints": ((16, 4), (32, 2), (64, 1))},
        "ours": {
            "initial": int(frozen["M"]),
            "checkpoints": ((int(frozen["checkpoint"]), int(frozen["K"])), (64, 1)),
        },
    }


def load_prompts() -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (EXP / "prompts_geneval_all_553.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 553 and [int(row["prompt_id"]) for row in rows] == list(range(553))
    return rows


def logical_evals(spec: dict[str, Any]) -> int:
    alive = int(spec["initial"])
    previous = total = 0
    for step, keep in spec["checkpoints"]:
        total += alive * (int(step) - previous)
        previous, alive = int(step), int(keep)
    return total


def build_pipeline():
    pipe = FKDStableDiffusion.from_pretrained(MODEL, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def explicit_pool(pipe, prompt_id: int, pool_size: int, base_seed: int):
    seeds = [base_seed + prompt_id * 32 + candidate_id for candidate_id in range(pool_size)]
    generators = [torch.Generator(device="cuda:0").manual_seed(seed) for seed in seeds]
    pool = pipe.prepare_latents(
        batch_size=pool_size,
        num_channels_latents=pipe.unet.config.in_channels,
        height=pipe.unet.config.sample_size * pipe.vae_scale_factor,
        width=pipe.unet.config.sample_size * pipe.vae_scale_factor,
        dtype=pipe.unet.dtype,
        device=torch.device("cuda:0"),
        generator=generators,
        latents=None,
    )
    return pool, seeds, [latent_hash(pool[index]) for index in range(pool_size)]


def apply_indices(tensor, indices: torch.Tensor, batch_size: int):
    if tensor is None:
        return None
    idx = indices.to(tensor.device)
    if tensor.shape[0] == batch_size * 2:
        return tensor[torch.cat([idx, idx + batch_size])]
    if tensor.shape[0] == batch_size:
        return tensor[idx]
    return tensor


def run_method(
    pipe,
    *,
    prompt_row: dict[str, Any],
    method: str,
    spec: dict[str, Any],
    pool: torch.Tensor,
    seeds: list[int],
    hashes: list[str],
    worker_index: int,
    repetition: int,
    base_seed: int,
) -> dict[str, Any]:
    prompt_id = int(prompt_row["prompt_id"])
    prompt = prompt_row["prompt"]
    initial = int(spec["initial"])
    checkpoints = dict(spec["checkpoints"])
    current_ids = list(range(initial))
    initial_latents = pool[:initial].clone()
    generators = [torch.Generator(device="cuda:0").manual_seed(seeds[index]) for index in range(initial)]
    trace = []
    online_reward_s = 0.0

    def callback(_pipe, step_index, timestep, kwargs):
        nonlocal current_ids, online_reward_s
        step = int(step_index) + 1
        if step not in checkpoints:
            return {"latents": kwargs["latents"]}
        torch.cuda.synchronize()
        score_started = time.perf_counter()
        decoded = latent_to_decode(model=pipe, output_type="pil", latents=kwargs["x0_preds"])
        scores = [
            float(value)
            for value in do_image_reward(prompts=[prompt] * len(current_ids), image_tensors=decoded)
        ]
        torch.cuda.synchronize()
        score_elapsed = time.perf_counter() - score_started
        online_reward_s += score_elapsed
        before = len(current_ids)
        keep = int(checkpoints[step])
        order = sorted(range(before), key=lambda index: (-scores[index], current_ids[index]))
        chosen = torch.tensor(order[:keep], device=kwargs["latents"].device, dtype=torch.long)
        survivors = [current_ids[index] for index in order[:keep]]
        outputs = {"latents": kwargs["latents"][chosen]}
        for key in ("prompt_embeds", "negative_prompt_embeds"):
            if key in kwargs:
                outputs[key] = apply_indices(kwargs[key], chosen, before)
        trace.append(
            {
                "step": step,
                "timestep": int(timestep.item()),
                "batch_before": before,
                "batch_after": keep,
                "candidate_ids": list(current_ids),
                "candidate_seeds": [seeds[index] for index in current_ids],
                "image_reward": scores,
                "survivor_ids": survivors,
                "reward_and_decode_s": score_elapsed,
            }
        )
        current_ids = survivors
        del decoded
        return outputs

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = pipe(
            prompt=[prompt] * initial,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            eta=ETA,
            generator=generators,
            latents=initial_latents,
            output_type="latent",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "x0_preds", "prompt_embeds", "negative_prompt_embeds"],
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    with torch.inference_mode():
        final_tensor = latent_to_decode(model=pipe, output_type="pil", latents=output.images)
        final_image = pipe.image_processor.postprocess(final_tensor, output_type="pil")[0]
    winner_id = current_ids[0]
    final_row = trace[-1]
    final_ir = float(final_row["image_reward"][final_row["candidate_ids"].index(winner_id)])
    image_path = EXP / "outputs" / method / f"rep{repetition}" / f"{prompt_id:05d}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = image_path.with_suffix(".tmp.png")
    final_image.save(temporary)
    os.replace(temporary, image_path)
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "tag": prompt_row["tag"],
        "worker_index": worker_index,
        "method": method,
        "repetition": repetition,
        "base_seed": base_seed,
        "model": MODEL,
        "dtype": "torch.float16",
        "scheduler": type(pipe.scheduler).__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "steps": STEPS,
        "eta": ETA,
        "guidance_scale": GUIDANCE,
        "candidate_ids": list(range(initial)),
        "candidate_seeds": seeds[:initial],
        "initial_latent_hashes": hashes[:initial],
        "trace": trace,
        "winner_id": winner_id,
        "winner_seed": seeds[winner_id],
        "final_image_reward": final_ir,
        "logical_unet_evals": logical_evals(spec),
        "batched_verifier_calls": len(spec["checkpoints"]),
        "verifier_candidate_scores": sum(row["batch_before"] for row in trace),
        "elapsed_s": elapsed,
        "online_reward_s": online_reward_s,
        "peak_vram_gib": peak,
        "image_path": str(image_path),
    }


def complete(prompt_id: int, repetition: int, method: str, worker_index: int, expected_evals: int) -> bool:
    image = EXP / "outputs" / method / f"rep{repetition}" / f"{prompt_id:05d}.png"
    metadata = EXP / "metadata" / f"gpu{worker_index}" / f"{prompt_id:05d}_rep{repetition}_{method}.json"
    if not image.exists() or not metadata.exists():
        return False
    try:
        row = json.loads(metadata.read_text())
        return row["logical_unet_evals"] == expected_evals and row["winner_id"] is not None
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
    protocol = load_protocol()
    base_seeds = [int(value) for value in protocol["fresh_seed_bases"]]
    frozen = load_frozen()
    methods = schedules(frozen)
    assert logical_evals(methods["psp"]) == 256
    assert logical_evals(methods["ours"]) == frozen["logical_compute"] <= 256
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
        EXP / "metadata" / f"gpu{args.worker_index}" / "hardware.json",
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
    pool_size = max(8, int(frozen["M"]))
    for row in owned:
        prompt_id = int(row["prompt_id"])
        for repetition, base_seed in enumerate(base_seeds):
            pending = [
                method
                for method, spec in methods.items()
                if not (args.resume and complete(prompt_id, repetition, method, args.worker_index, logical_evals(spec)))
            ]
            if not pending:
                continue
            pool, seeds, hashes = explicit_pool(pipe, prompt_id, pool_size, base_seed)
            for method in pending:
                result = run_method(
                    pipe,
                    prompt_row=row,
                    method=method,
                    spec=methods[method],
                    pool=pool,
                    seeds=seeds,
                    hashes=hashes,
                    worker_index=args.worker_index,
                    repetition=repetition,
                    base_seed=base_seed,
                )
                atomic_json(
                    EXP / "metadata" / f"gpu{args.worker_index}" / f"{prompt_id:05d}_rep{repetition}_{method}.json",
                    result,
                )
                print(
                    json.dumps(
                        {
                            key: result[key]
                            for key in (
                                "prompt_id",
                                "repetition",
                                "base_seed",
                                "method",
                                "logical_unet_evals",
                                "batched_verifier_calls",
                                "verifier_candidate_scores",
                                "elapsed_s",
                                "peak_vram_gib",
                                "final_image_reward",
                            )
                        }
                    ),
                    flush=True,
                )
        completed += 1
    print(json.dumps({"worker": args.worker_index, "completed": completed, "owned": len(owned)}), flush=True)


if __name__ == "__main__":
    main()
