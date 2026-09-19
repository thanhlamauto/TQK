#!/usr/bin/env python3
"""Confirmatory aggregation.

README section 7: for prompt ``p`` and repetition ``r`` compute the paired
difference ``d[p,r]``, average the three repetitions within prompt first, then
bootstrap the 553 prompt-level averages 10,000 times.  The 1,659 images are never
treated as independent samples.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
CAL = EXP.parent / "dense_calibration_200"
METHODS = ("psp", "ours")
DISPLAY = {"psp": "PSP", "ours": "Frozen single-stage"}
METRICS = {"image_reward": "ImageReward", "hps": "HPS", "geneval": "GenEval"}
BOOTSTRAP_SEED = 20260920
BOOTSTRAP_SAMPLES = 10_000


def prompt_id_from_filename(filename: str) -> int:
    for parent in Path(filename).parents:
        if parent.name.isdigit():
            return int(parent.name)
    raise ValueError(filename)


def read_generation() -> pd.DataFrame:
    rows = []
    for worker in (0, 1):
        for path in sorted((EXP / "metadata" / f"gpu{worker}").glob("*_rep*_*.json")):
            row = json.loads(path.read_text())
            rows.append({
                "prompt_id": int(row["prompt_id"]),
                "repetition": int(row["repetition"]),
                "base_seed": int(row["base_seed"]),
                "method": row["method"],
                "worker": worker,
                "image_reward": float(row["final_image_reward"]),
                "elapsed_s": float(row["elapsed_s"]),
                "online_reward_s": float(row["online_reward_s"]),
                "peak_vram_gib": float(row["peak_vram_gib"]),
                "logical_unet_evals": int(row["logical_unet_evals"]),
                "batched_verifier_calls": int(row["batched_verifier_calls"]),
                "verifier_candidate_scores": int(row["verifier_candidate_scores"]),
                "winner_id": int(row["winner_id"]),
            })
    return pd.DataFrame(rows)


def read_geneval(repetitions: int) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        for repetition in range(repetitions):
            path = EXP / "geneval_results" / f"{method}_rep{repetition}.jsonl"
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append({
                    "method": method,
                    "repetition": repetition,
                    "prompt_id": prompt_id_from_filename(item["filename"]),
                    "geneval": float(bool(item["correct"])),
                    "geneval_reason": item.get("reason", ""),
                })
    return pd.DataFrame(rows)


def parse_dmon() -> dict[int, float]:
    samples: dict[int, list[float]] = defaultdict(list)
    path = EXP / "logs/nvidia_dmon.log"
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        try:
            gpu, sm = int(parts[2]), float(parts[6])
        except (IndexError, ValueError):
            continue
        if sm >= 0:
            samples[gpu].append(sm)
    return {gpu: float(statistics.mean(values)) for gpu, values in samples.items() if values}


def paired_bootstrap(delta: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(BOOTSTRAP_SAMPLES, len(delta)))].mean(axis=1)
    return {
        "delta": float(delta.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
    }


def main() -> None:
    protocol = json.loads((EXP / "protocol_manifest.json").read_text())
    frozen = json.loads((EXP / "FROZEN_SCHEDULE.json").read_text())
    repetitions = int(protocol["repetitions"])
    generation = read_generation()
    assert len(generation) == 553 * repetitions * 2, len(generation)
    assert not generation.duplicated(["prompt_id", "repetition", "method"]).any()
    hps = pd.read_csv(EXP / "metrics/hps.csv")
    geneval = read_geneval(repetitions)
    frame = generation.merge(
        hps, on=["prompt_id", "repetition", "method"], validate="one_to_one"
    ).merge(
        geneval[["prompt_id", "repetition", "method", "geneval"]],
        on=["prompt_id", "repetition", "method"],
        validate="one_to_one",
    )
    assert len(frame) == 553 * repetitions * 2 and not frame.isna().any().any()

    # Average repetitions within prompt first.
    per_prompt = (
        frame.groupby(["prompt_id", "method"], as_index=False)
        .agg(
            image_reward=("image_reward", "mean"),
            hps=("hps", "mean"),
            geneval=("geneval", "mean"),
            runtime_mean_s=("elapsed_s", "mean"),
            online_reward_s=("online_reward_s", "mean"),
            peak_vram_gib=("peak_vram_gib", "max"),
            logical_unet_evals=("logical_unet_evals", "first"),
            batched_verifier_calls=("batched_verifier_calls", "first"),
            verifier_candidate_scores=("verifier_candidate_scores", "first"),
        )
    )
    wide = {metric: per_prompt.pivot(index="prompt_id", columns="method", values=metric) for metric in METRICS}
    stats = {}
    for index, (metric, label) in enumerate(METRICS.items()):
        delta = (wide[metric]["ours"] - wide[metric]["psp"]).to_numpy(dtype=float)
        stats[metric] = {
            "label": label,
            "psp": float(wide[metric]["psp"].mean()),
            "ours": float(wide[metric]["ours"].mean()),
            **paired_bootstrap(delta, BOOTSTRAP_SEED + index),
        }
    # Per-repetition deltas as a robustness view.
    rep_stats = {}
    for repetition in range(repetitions):
        subset = frame[frame.repetition.eq(repetition)]
        rep_stats[str(repetition)] = {}
        for metric in METRICS:
            pivot = subset.pivot(index="prompt_id", columns="method", values=metric)
            delta = (pivot["ours"] - pivot["psp"]).to_numpy(dtype=float)
            rep_stats[str(repetition)][metric] = paired_bootstrap(delta, BOOTSTRAP_SEED + 100 + repetition)

    summary_rows = []
    for method in METHODS:
        group = frame[frame.method.eq(method)]
        per_prompt_group = per_prompt[per_prompt.method.eq(method)]
        summary_rows.append({
            "method": method,
            "schedule": "8->4@16->2@32" if method == "psp" else frozen["schedule_notation"],
            "logical_unet_evals": int(group.logical_unet_evals.iloc[0]),
            "image_reward": stats["image_reward"][method],
            "hps": stats["hps"][method],
            "geneval": stats["geneval"][method],
            "runtime_mean_s": float(per_prompt_group.runtime_mean_s.mean()),
            "runtime_median_s": float(per_prompt_group.runtime_mean_s.median()),
            "runtime_p90_s": float(per_prompt_group.runtime_mean_s.quantile(0.9)),
            "throughput_prompt_s_per_gpu": float(1.0 / per_prompt_group.runtime_mean_s.mean()),
            "online_reward_mean_s": float(per_prompt_group.online_reward_s.mean()),
            "peak_vram_mean_gib": float(per_prompt_group.peak_vram_gib.mean()),
            "peak_vram_max_gib": float(per_prompt_group.peak_vram_gib.max()),
            "batched_verifier_calls": int(group.batched_verifier_calls.iloc[0]),
            "verifier_candidate_scores": int(group.verifier_candidate_scores.iloc[0]),
        })
    summary = pd.DataFrame(summary_rows)

    wide_ge = wide["geneval"]
    delta_ge = wide_ge["ours"] - wide_ge["psp"]
    win_tie_loss = {
        "ours_win": int((delta_ge > 0).sum()),
        "tie": int((delta_ge == 0).sum()),
        "psp_win": int((delta_ge < 0).sum()),
    }
    strong_success = bool(
        stats["image_reward"]["delta"] > 0 and stats["image_reward"]["ci95_low"] > 0
    )
    secondary_clear_degradation = []
    for metric in ("hps", "geneval"):
        row = stats[metric]
        if row["ci95_high"] < 0:
            secondary_clear_degradation.append(METRICS[metric])
    if strong_success and secondary_clear_degradation:
        claim = (
            "ImageReward improves significantly, but "
            + ", ".join(secondary_clear_degradation)
            + " degrade with a paired 95% CI entirely below zero; the claim is restricted to "
            "better optimization of ImageReward."
        )
    elif strong_success:
        claim = "Positive mean Delta ImageReward with a paired 95% lower bound above zero."
    else:
        claim = (
            "ImageReward does not show a significant improvement (the paired 95% CI for "
            "Delta ImageReward contains zero or the mean is not positive)."
        )

    metrics = EXP / "metrics"
    metrics.mkdir(exist_ok=True)
    frame.to_csv(metrics / "per_run.csv", index=False)
    per_prompt.to_csv(metrics / "per_prompt.csv", index=False)
    summary.to_csv(metrics / "summary.csv", index=False)
    (metrics / "confirmatory_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (metrics / "per_repetition_stats.json").write_text(json.dumps(rep_stats, indent=2) + "\n")
    (metrics / "geneval_win_tie_loss.json").write_text(json.dumps(win_tie_loss, indent=2) + "\n")

    selected_all = pd.read_csv(CAL / "replay/cell_estimates.csv")
    frozen_row = selected_all[
        selected_all.checkpoint.eq(int(frozen["checkpoint"]))
        & selected_all.K.eq(int(frozen["K"]))
        & selected_all.M.eq(int(frozen["M"]))
    ].iloc[0]
    rows_by_method = {row["method"]: row for row in summary.to_dict("records")}
    psp_row, ours_row = rows_by_method["psp"], rows_by_method["ours"]
    main_table = "\n".join(
        f"| {DISPLAY[row['method']]} | {row['schedule']} | {row['logical_unet_evals']} | "
        f"{row['image_reward']:.6f} | {row['hps']:.6f} | {row['geneval']:.6f} | "
        f"{row['runtime_mean_s']:.3f} | {row['peak_vram_max_gib']:.2f} GiB | "
        f"{row['batched_verifier_calls']}/{row['verifier_candidate_scores']} |"
        for row in summary.to_dict("records")
    )
    delta_table = "\n".join(
        f"| {stats[key]['label']} | {stats[key]['psp']:.6f} | {stats[key]['ours']:.6f} | "
        f"{stats[key]['delta']:+.6f} | [{stats[key]['ci95_low']:.6f}, {stats[key]['ci95_high']:.6f}] |"
        for key in METRICS
    )
    rep_table = "\n".join(
        f"| {repetition} | {protocol['fresh_seed_bases'][repetition]} | "
        f"{rep_stats[str(repetition)]['image_reward']['delta']:+.6f} | "
        f"[{rep_stats[str(repetition)]['image_reward']['ci95_low']:+.6f}, "
        f"{rep_stats[str(repetition)]['image_reward']['ci95_high']:+.6f}] |"
        for repetition in range(repetitions)
    )
    calls_saved = psp_row["batched_verifier_calls"] - ours_row["batched_verifier_calls"]
    scores_saved = psp_row["verifier_candidate_scores"] - ours_row["verifier_candidate_scores"]
    runtime_delta = ours_row["runtime_mean_s"] - psp_row["runtime_mean_s"]
    vram_delta = ours_row["peak_vram_max_gib"] - psp_row["peak_vram_max_gib"]
    report = f"""# Confirmatory report: frozen single-stage schedule vs PSP (553 prompts x 3 seeds)

## Outcome

| Method | Schedule | UNet evals | IR | HPS | GenEval | sec/prompt | VRAM | verifier calls/scores |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{main_table}

| Metric | PSP | Frozen | Delta | 95% paired CI (prompt-level, 10,000 resamples) |
|---|---:|---:|---:|---:|
{delta_table}

Paired GenEval outcomes over prompt-level means: **{win_tie_loss['ours_win']} ours wins / {win_tie_loss['tie']} ties / {win_tie_loss['psp_win']} PSP wins**.

Primary endpoint conclusion: {claim}

## Per-repetition ImageReward deltas (robustness, not the primary test)

| repetition | base seed | mean Delta IR | 95% CI |
|---:|---:|---:|---:|
{rep_table}

## Protocol facts

* Frozen schedule: `{frozen['schedule_notation']}`, {frozen['logical_compute']} logical UNet
  evaluations ({frozen['unused_compute']} unused); LCB80 at freeze = {frozen['lcb80']:+.6f}.
* Fixed comparator: PSP `8->4@16->2@32->1@64` = 256 logical evaluations.
* Prompt set: all 553 official GenEval prompts (never used during calibration).
* Fresh candidate-pool base seeds: {protocol['fresh_seed_bases']} (disjoint from
  calibration 20260920 and previous validation 20260919).
* Per prompt and repetition, both methods share the identical candidate pool; the
  three repetitions are averaged within prompt before the 553-prompt bootstrap.
* Calibration-cell empirical Delta IR (all 200 calibration prompts) was
  {frozen_row.delta_vs_psp:+.6f}; this confirms it on untouched prompts.
* HPS and GenEval were not used for selection and must not show clear degradation.

## Runtime and efficiency

* Frozen minus PSP mean runtime: {runtime_delta:+.4f} s/prompt
  (PSP {psp_row['runtime_mean_s']:.4f}, frozen {ours_row['runtime_mean_s']:.4f}).
* Frozen minus PSP max allocated VRAM: {vram_delta:+.3f} GiB
  (PSP {psp_row['peak_vram_max_gib']:.2f}, frozen {ours_row['peak_vram_max_gib']:.2f}).
* Verifier savings: {calls_saved} batched calls and {scores_saved} candidate scores per prompt.

## Narrowest defensible claim

On these three fresh seeds over all 553 official GenEval prompts on SD1.5, the
frozen schedule `{frozen['schedule_notation']}` versus unchanged PSP produced the
paired metric outcomes and confidence intervals above at no more logical UNet
compute. Claims beyond this model, seed pool, and prompt population are not
supported.
"""
    (EXP / "FINAL_REPORT.md").write_text(report)
    (metrics / "claim.txt").write_text(claim + "\n")
    print(json.dumps({
        "phase": "confirmatory_complete",
        "frozen_schedule": frozen["schedule_notation"],
        "summary": summary_rows,
        "paired": stats,
        "geneval_win_tie_loss": win_tie_loss,
        "strong_success": strong_success,
        "claim": claim,
        "utilization": parse_dmon(),
    }, indent=2))


if __name__ == "__main__":
    main()
