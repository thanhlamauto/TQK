#!/usr/bin/env python3
"""Dense calibration analysis, out-of-fold gate, and LCB80 freeze.

Implements README sections 3-6:

* breadth value ``O(M)`` via the exact expected-oracle formula;
* subset-draw estimates of ``p_miss``, ``ell``, ``R`` and ``Q``;
* low-complexity shared reliability surfaces (``reliability_models.py``);
* deterministic five-fold out-of-fold selection;
* prompt bootstrap 80% lower confidence bound freeze gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from policies import (
    BANK_CHECKPOINTS,
    FINAL_STEP,
    PSP,
    PSP_LOGICAL_COMPUTE,
    PSP_NOTATION,
    STEPS,
    as_dict,
    valid_policies,
)
from reliability_models import fit_ell, fit_miss, predict_miss_ell

EXP = Path(__file__).resolve().parent
PROMPTS = EXP / "prompts/calibration_prompts.jsonl"
MANIFEST = EXP / "prompts/prompt_manifest.json"
REPLAY = EXP / "replay"

SUBSET_SEED = 20260920
SUBSETS_PER_PROMPT = 500
BOOTSTRAP_SEED = 20260920
BOOTSTRAP_RESAMPLES = 1000
LCB_QUANTILE = 0.10  # 80% lower confidence bound
GATE_MIN_POSITIVE_FOLDS = 4
GATE_MIN_RANK_CORRELATION = 0.80
GATE_COHERENT_CHECKPOINT_SPAN = 4
GATE_COHERENT_K_SPAN = 1
EXPECTED_GENEVAL_OVERLAP = 0
BANK_PROMPT_COUNT = 200
CANDIDATES = 25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompts() -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in PROMPTS.read_text().splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows).sort_values("prompt_id").reset_index(drop=True)
    assert len(frame) == BANK_PROMPT_COUNT
    assert frame.prompt_id.tolist() == list(range(BANK_PROMPT_COUNT))
    return frame


def load_bank(prompts: pd.DataFrame) -> dict:
    ckpt_index = {step: index for index, step in enumerate(BANK_CHECKPOINTS)}
    n = len(prompts)
    y = np.full((n, CANDIDATES), np.nan)
    x = np.full((n, CANDIDATES, len(BANK_CHECKPOINTS)), np.nan)
    for position, prompt_id in enumerate(prompts.prompt_id):
        candidates = None
        for worker in (0, 1):
            path = EXP / "bank_raw" / f"gpu{worker}" / f"{prompt_id:05d}.json"
            if path.exists():
                candidates = json.loads(path.read_text())["candidates"]
                break
        assert candidates is not None, prompt_id
        candidates = sorted(candidates, key=lambda row: row["candidate_id"])
        assert [row["candidate_id"] for row in candidates] == list(range(CANDIDATES))
        for candidate in candidates:
            cid = int(candidate["candidate_id"])
            y[position, cid] = float(candidate["IR_final"])
            for step in BANK_CHECKPOINTS:
                x[position, cid, ckpt_index[step]] = float(candidate[f"IR_t{step}"])
    assert np.isfinite(y).all() and np.isfinite(x).all()
    return {"y": y, "x": x, "ckpt_index": ckpt_index, "fold": prompts.fold.to_numpy(dtype=int)}


def expected_oracle(y_row: np.ndarray, pool: int) -> float:
    """Exact E[max of a uniformly random ``pool``-subset] from the 25 finals."""
    ordered = np.sort(y_row)  # ascending: Y_(1) <= ... <= Y_(25)
    n = len(ordered)
    denominator = math.comb(n, pool)
    total = 0.0
    for j in range(pool, n + 1):  # j is 1-indexed rank
        numerator = math.comb(j - 1, pool - 1)
        total += float(ordered[j - 1]) * numerator / denominator
    return total


def subset_draws(rng: np.random.Generator, pool: int) -> np.ndarray:
    # Independent uniform subsets without replacement: rank a random score matrix
    # along each row and take the first ``pool`` columns.
    ranks = np.argsort(rng.random((SUBSETS_PER_PROMPT, CANDIDATES)), axis=1)
    return ranks[:, :pool].astype(np.int64)


def replay_psp_on_subsets(y_prompt: np.ndarray, x_prompt: np.ndarray, ckpt_index, subsets: np.ndarray) -> np.ndarray:
    """Fixed PSP 8->4@16->2@32->1@64 final winner value per subset."""
    values = np.take_along_axis(y_prompt[None, :], subsets, axis=1)  # (S, 8)
    x16 = np.take_along_axis(x_prompt[None, :, ckpt_index[16]], subsets, axis=1)
    order16 = np.lexsort((subsets, -x16), axis=1)[:, :4]
    ids16 = np.take_along_axis(subsets, order16, axis=1)
    x32 = np.take_along_axis(x_prompt[None, :, ckpt_index[32]], ids16, axis=1)
    order32 = np.lexsort((ids16, -x32), axis=1)[:, :2]
    ids2 = np.take_along_axis(ids16, order32, axis=1)
    y2 = np.take_along_axis(y_prompt[None, :], ids2, axis=1)
    return y2.max(axis=1)


def collect_subset_statistics(bank: dict, policies: list) -> dict:
    y, x = bank["y"], bank["x"]
    ckpt_index = bank["ckpt_index"]
    n = y.shape[0]
    n_policies = len(policies)
    miss = np.zeros((n, n_policies))
    regret_sum = np.zeros((n, n_policies))
    miss_regret_sum = np.zeros((n, n_policies))
    pool_sizes = sorted({policy.M for policy in policies} | {PSP["initial"]})
    oracle_curve = {pool: np.zeros(n) for pool in pool_sizes}
    psp_q = np.zeros(n)

    for prompt_position in range(n):
        for pool in pool_sizes:
            oracle_curve[pool][prompt_position] = expected_oracle(y[prompt_position], pool)

    for pool in pool_sizes:
        policy_indices = [index for index, policy in enumerate(policies) if policy.M == pool]
        for prompt_position in range(n):
            rng = np.random.default_rng([SUBSET_SEED, pool, prompt_position])
            subsets = subset_draws(rng, pool)
            y_prompt = y[prompt_position]
            y_sub = np.take_along_axis(y_prompt[None, :], subsets, axis=1)  # (S, pool)
            final_order = np.lexsort((subsets, -y_sub), axis=1)
            best_position = final_order[:, :1]
            oracle = np.take_along_axis(y_sub, best_position, axis=1)[:, 0]
            best_id = np.take_along_axis(subsets, best_position, axis=1)[:, 0]
            if pool == PSP["initial"]:
                psp_q[prompt_position] = float(replay_psp_on_subsets(y_prompt, x[prompt_position], ckpt_index, subsets).mean())
            for policy_index in policy_indices:
                policy = policies[policy_index]
                x_sub = np.take_along_axis(
                    x[prompt_position][None, :, ckpt_index[policy.checkpoint]],
                    subsets,
                    axis=1,
                )
                order = np.lexsort((subsets, -x_sub), axis=1)[:, : policy.K]
                survivor_ids = np.take_along_axis(subsets, order, axis=1)
                survivor_y = np.take_along_axis(y_sub, order, axis=1).max(axis=1)
                regret = oracle - survivor_y
                is_miss = ~np.any(survivor_ids == best_id[:, None], axis=1)
                miss[prompt_position, policy_index] = float(is_miss.sum())
                regret_sum[prompt_position, policy_index] = float(regret.sum())
                miss_regret_sum[prompt_position, policy_index] = float(regret[is_miss].sum())
    return {
        "miss": miss,
        "regret_sum": regret_sum,
        "miss_regret_sum": miss_regret_sum,
        "oracle_curve": oracle_curve,
        "psp_q": psp_q,
    }


def aggregate_over_prompts(stats: dict, policies: list, oracle_curve: dict, indices: np.ndarray) -> dict:
    n_tot = len(indices) * SUBSETS_PER_PROMPT
    miss_count = stats["miss"][indices, :].sum(axis=0)
    n_miss = miss_count
    p_miss = miss_count / n_tot
    regret = stats["regret_sum"][indices, :].sum(axis=0) / n_tot
    with np.errstate(invalid="ignore", divide="ignore"):
        ell = np.where(n_miss > 0, stats["miss_regret_sum"][indices, :].sum(axis=0) / np.maximum(n_miss, 1.0), np.nan)
    o_pool = {pool: float(values[indices].mean()) for pool, values in oracle_curve.items()}
    o = np.array([o_pool[policy.M] for policy in policies])
    q_psp = float(stats["psp_q"][indices].mean())
    return {
        "p_miss": p_miss,
        "ell": ell,
        "regret": regret,
        "n_miss": n_miss,
        "n_total": n_tot,
        "o_pool": o_pool,
        "o": o,
        "q_psp": q_psp,
        "q": o - regret,
        "delta": o - regret - q_psp,
    }


def feature_arrays(policies: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.array([policy.q for policy in policies])
    log_m = np.log(np.array([policy.M for policy in policies], dtype=float))
    ratio = np.array([policy.ratio for policy in policies])
    return q, log_m, ratio


def fit_surfaces(policies: list, agg: dict) -> tuple:
    q, log_m, ratio = feature_arrays(policies)
    valid = np.isfinite(agg["ell"])
    miss_model = fit_miss(q, log_m, ratio, agg["p_miss"], agg["n_total"])
    ell_model = fit_ell(
        q[valid],
        log_m[valid],
        ratio[valid],
        agg["ell"][valid],
        np.maximum(agg["n_miss"][valid], 1.0),
    )
    return miss_model, ell_model


def predicted_delta(policies: list, miss_model, ell_model, agg: dict) -> np.ndarray:
    q, log_m, ratio = feature_arrays(policies)
    p_miss, ell = predict_miss_ell(miss_model, ell_model, q, log_m, ratio)
    return agg["o"] - p_miss * ell - agg["q_psp"]


def cell_table(policies: list, agg: dict) -> pd.DataFrame:
    rows = []
    for index, policy in enumerate(policies):
        rows.append(
            {
                **as_dict(policy),
                "q": policy.q,
                "ratio": policy.ratio,
                "p_miss": float(agg["p_miss"][index]),
                "ell": float(agg["ell"][index]),
                "regret_mean": float(agg["regret"][index]),
                "p_miss_times_ell": float(agg["p_miss"][index] * agg["ell"][index]),
                "O_M": float(agg["o"][index]),
                "Q_empirical": float(agg["q"][index]),
                "delta_vs_psp": float(agg["delta"][index]),
                "n_miss": int(agg["n_miss"][index]),
            }
        )
    return pd.DataFrame(rows)


def oof_analysis(policies: list, stats: dict, oracle_curve: dict, fold: np.ndarray) -> tuple[pd.DataFrame, dict]:
    folds = sorted(int(value) for value in np.unique(fold))
    q, log_m, ratio = feature_arrays(policies)
    records = []
    for held_fold in folds:
        train = np.where(fold != held_fold)[0]
        held = np.where(fold == held_fold)[0]
        train_agg = aggregate_over_prompts(stats, policies, oracle_curve, train)
        held_agg = aggregate_over_prompts(stats, policies, oracle_curve, held)
        miss_model, ell_model = fit_surfaces(policies, train_agg)
        predicted_train = predicted_delta(policies, miss_model, ell_model, train_agg)
        predicted_held = predicted_delta(policies, miss_model, ell_model, held_agg)
        selected_index = int(np.argmax(predicted_train))
        actual_held = held_agg["delta"]
        selected_actual = float(actual_held[selected_index])
        rho = float(spearmanr(predicted_held, actual_held).statistic)
        actual_order = np.argsort(-actual_held)
        predicted_winner = int(np.argmax(predicted_held))
        predicted_winner_rank = int(np.where(actual_order == predicted_winner)[0][0]) + 1
        actual_top3 = set(actual_order[:3].tolist())
        actual_top5 = set(actual_order[:5].tolist())
        per_prompt_pair = (
            held_agg["o"][selected_index] - held_agg["regret"][selected_index]
        )
        # paired per-prompt delta for the selected policy
        prompt_aggregate = stats["regret_sum"][held, selected_index] / SUBSETS_PER_PROMPT
        per_prompt = (
            oracle_curve[policies[selected_index].M][held]
            - prompt_aggregate
            - stats["psp_q"][held]
        )
        records.append(
            {
                "fold": held_fold,
                "n_train": len(train),
                "n_held": len(held),
                "selected_policy": policies[selected_index].notation,
                "selected_checkpoint": policies[selected_index].checkpoint,
                "selected_K": policies[selected_index].K,
                "selected_M": policies[selected_index].M,
                "selected_predicted_train_delta": float(predicted_train[selected_index]),
                "selected_held_delta": selected_actual,
                "selected_held_paired_mean": float(per_prompt.mean()),
                "selected_held_paired_se": float(per_prompt.std(ddof=1) / math.sqrt(len(per_prompt))),
                "positive": selected_actual > 0,
                "rank_spearman_predicted_vs_actual": rho,
                "predicted_winner": policies[predicted_winner].notation,
                "predicted_winner_actual_rank": predicted_winner_rank,
                "predicted_winner_in_actual_top3": predicted_winner in actual_top3,
                "predicted_winner_in_actual_top5": predicted_winner in actual_top5,
                "train_agg_o_check": per_prompt_pair,
            }
        )
    frame = pd.DataFrame(records)
    selected_checkpoints = frame.selected_checkpoint.to_numpy(dtype=int)
    selected_ks = frame.selected_K.to_numpy(dtype=int)
    coherent = bool(
        selected_checkpoints.max() - selected_checkpoints.min() <= GATE_COHERENT_CHECKPOINT_SPAN
        and selected_ks.max() - selected_ks.min() <= GATE_COHERENT_K_SPAN
    )
    gate = {
        "positive_mean_held_delta": bool(frame.selected_held_delta.mean() > 0),
        "mean_held_delta": float(frame.selected_held_delta.mean()),
        "positive_folds": int(frame.positive.sum()),
        "min_positive_folds": GATE_MIN_POSITIVE_FOLDS,
        "positive_fold_gate": bool(frame.positive.sum() >= GATE_MIN_POSITIVE_FOLDS),
        "mean_spearman": float(frame.rank_spearman_predicted_vs_actual.mean()),
        "min_rank_correlation": GATE_MIN_RANK_CORRELATION,
        "rank_correlation_gate": bool(frame.rank_spearman_predicted_vs_actual.mean() >= GATE_MIN_RANK_CORRELATION),
        "coherent_neighborhood": coherent,
        "selected_checkpoint_span": int(selected_checkpoints.max() - selected_checkpoints.min()),
        "selected_K_span": int(selected_ks.max() - selected_ks.min()),
        "predicted_winner_in_top3_folds": int(frame.predicted_winner_in_actual_top3.sum()),
        "predicted_winner_in_top5_folds": int(frame.predicted_winner_in_actual_top5.sum()),
    }
    gate["pass"] = bool(
        gate["positive_mean_held_delta"]
        and gate["positive_fold_gate"]
        and gate["rank_correlation_gate"]
        and gate["coherent_neighborhood"]
    )
    return frame, gate


def bootstrap_freeze(policies: list, stats: dict, oracle_curve: dict) -> pd.DataFrame:
    n_prompts = stats["psp_q"].shape[0]
    indices_all = np.arange(n_prompts)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n_prompts, size=(BOOTSTRAP_RESAMPLES, n_prompts))
    deltas = np.zeros((BOOTSTRAP_RESAMPLES, len(policies)))
    for resample_index, indices in enumerate(draws):
        agg = aggregate_over_prompts(stats, policies, oracle_curve, indices)
        miss_model, ell_model = fit_surfaces(policies, agg)
        deltas[resample_index] = predicted_delta(policies, miss_model, ell_model, agg)
    full = aggregate_over_prompts(stats, policies, oracle_curve, indices_all)
    miss_model, ell_model = fit_surfaces(policies, full)
    full_predicted = predicted_delta(policies, miss_model, ell_model, full)
    rows = []
    for index, policy in enumerate(policies):
        lcb = float(np.quantile(deltas[:, index], LCB_QUANTILE))
        rows.append(
            {
                **as_dict(policy),
                "q": policy.q,
                "ratio": policy.ratio,
                "empirical_delta": float(full["delta"][index]),
                "predicted_delta": float(full_predicted[index]),
                "bootstrap_mean": float(deltas[:, index].mean()),
                "lcb80": lcb,
                "ucb80": float(np.quantile(deltas[:, index], 1.0 - LCB_QUANTILE)),
                "ci95_low": float(np.quantile(deltas[:, index], 0.025)),
                "ci95_high": float(np.quantile(deltas[:, index], 0.975)),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("lcb80", ascending=False).reset_index(drop=True)


def write_report(policies, stats, full_agg, cells, oof, gate, bootstrap, frozen, overlap_note) -> None:
    invariant = float(np.max(np.abs(cells.regret_mean - cells.p_miss_times_ell)))
    best_cell = cells.sort_values("delta_vs_psp", ascending=False).iloc[0]
    top_bootstrap = bootstrap.head(5)
    oof_table = "\n".join(
        f"| {row.fold} | {row.selected_policy} | {row.selected_held_delta:+.6f} | "
        f"{'yes' if row.positive else 'no'} | {row.rank_spearman_predicted_vs_actual:.3f} | "
        f"{row.predicted_winner_actual_rank} | {row.predicted_winner} |"
        for row in oof.itertuples()
    )
    lcb_table = "\n".join(
        f"| {row.notation} | {row.predicted_delta:+.6f} | {row.lcb80:+.6f} | "
        f"{row.ci95_low:+.6f} | {row.ci95_high:+.6f} |"
        for row in top_bootstrap.itertuples()
    )
    if frozen:
        frozen_block = (
            f"Frozen schedule: **{frozen['schedule_notation']}** "
            f"(LCB80 = {frozen['lcb80']:+.6f}, predicted Delta IR = {frozen['predicted_delta']:+.6f})."
        )
    elif not gate["pass"]:
        frozen_block = (
            "Out-of-fold gate **FAILED**; per the pre-registered stop rule the confirmatory "
            "553-prompt phase was not run."
        )
    else:
        frozen_block = (
            "Freeze gate **FAILED**: no policy reached a positive 80% lower confidence bound; "
            "the confirmatory 553-prompt phase was not run."
        )
    report = f"""# Dense single-stage calibration report (README sections 3-6)

## Outcome

{frozen_block}

* Calibration corpus: 200 fixed prompts from the repository ImageReward
  `test_ir.json`, selection seed 20260920, {overlap_note}
  Zero exact overlap with the 553 GenEval prompts.
* Search space: exactly 57 policies ``t in {{10..28}}``, ``K in {{1,2,3}}``, with
  ``M(t,K)=floor((256-K*(64-t))/t)``. ``M`` was never tuned independently.
* Bank: 200 prompts x 25 independent trajectories x
  {len(BANK_CHECKPOINTS)} checkpoints (10..28 plus 32 for the fixed PSP replay) plus
  the final step, scored with ImageReward on the scheduler's own
  ``pred_original_sample``.

## Section 3 - breadth value and pruning regret

* Invariant ``max |mean regret - p_miss * ell|`` = {invariant:.3e}.
* Best empirical cell (all 200 prompts): **{best_cell.notation}** with
  Delta IR {best_cell.delta_vs_psp:+.6f}.
* Fixed PSP reference: ``{PSP_NOTATION}`` = {PSP_LOGICAL_COMPUTE} logical UNet
  evaluations; mean calibration IR {full_agg['q_psp']:.6f}.

## Section 4 - shared reliability surfaces

* ``p_miss``: logistic with hinge basis in ``q=t/64`` and ``w_j >= 0`` (monotone
  non-increasing in denoising depth), plus ``log(M)`` and ``K/M``; L2 = 1e-3.
* ``ell``: Gamma regression with log link on ``[1, log(M), K/M, q, q^2]``; L2 = 1e-3.
* Full-200 fitted surfaces produced the bootstrap table below.

## Section 5 - five-fold out-of-fold selection

| fold | selected | held Delta IR | positive | Spearman | predicted winner rank | predicted winner |
|---:|---|---:|:--:|---:|---:|---|
{oof_table}

| gate | value | threshold | pass |
|---|---:|---:|:--:|
| mean held Delta IR > 0 | {gate['mean_held_delta']:+.6f} | > 0 | {gate['positive_mean_held_delta']} |
| positive folds | {gate['positive_folds']} | >= {gate['min_positive_folds']} | {gate['positive_fold_gate']} |
| mean rank Spearman | {gate['mean_spearman']:.3f} | >= {gate['min_rank_correlation']} | {gate['rank_correlation_gate']} |
| coherent neighborhood | span t={gate['selected_checkpoint_span']}, span K={gate['selected_K_span']} | t<={GATE_COHERENT_CHECKPOINT_SPAN}, K<={GATE_COHERENT_K_SPAN} | {gate['coherent_neighborhood']} |
| **overall OOF gate** |  |  | **{gate['pass']}** |

Predicted winner was in the actual top-3 in {gate['predicted_winner_in_top3_folds']}/5 folds and
top-5 in {gate['predicted_winner_in_top5_folds']}/5 folds.

## Section 6 - freeze gate (prompt bootstrap, {BOOTSTRAP_RESAMPLES} resamples)

Top policies by 80% lower confidence bound of predicted Delta IR:

| policy | predicted Delta IR | LCB80 | CI95 low | CI95 high |
|---|---:|---:|---:|---:|
{lcb_table}

## Interpretation guardrails

* This is calibration evidence only. HPS and official GenEval were not inspected
  during calibration and selection.
* The confirmatory endpoint (README section 7) is a separate 553-prompt, three
  fresh-seed comparison and is run only if this freeze gate passes.
"""
    (EXP / "CALIBRATION_REPORT.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    REPLAY.mkdir(exist_ok=True)
    prompts = load_prompts()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["exact_geneval_prompt_overlap"] == EXPECTED_GENEVAL_OVERLAP
    bank = load_bank(prompts)
    policies = valid_policies()
    assert len(policies) == 57
    stats = collect_subset_statistics(bank, policies)
    all_indices = np.arange(len(prompts))
    full_agg = aggregate_over_prompts(stats, policies, stats["oracle_curve"], all_indices)
    cells = cell_table(policies, full_agg)
    cells.to_csv(REPLAY / "cell_estimates.csv", index=False)
    invariant = float(np.max(np.abs(cells.regret_mean - cells.p_miss_times_ell)))
    assert invariant < 1e-9, invariant

    oof_frame, gate = oof_analysis(policies, stats, stats["oracle_curve"], bank["fold"])
    oof_frame.to_csv(REPLAY / "oof_folds.csv", index=False)
    (REPLAY / "oof_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    frozen = None
    bootstrap = pd.DataFrame()
    if gate["pass"] and not args.skip_bootstrap:
        bootstrap = bootstrap_freeze(policies, stats, stats["oracle_curve"])
        bootstrap.to_csv(REPLAY / "bootstrap_lcb.csv", index=False)
        winner = bootstrap.iloc[0]
        if float(winner.lcb80) > 0:
            frozen = {
                "checkpoint": int(winner.checkpoint),
                "K": int(winner.K),
                "M": int(winner.M),
                "logical_compute": int(winner.logical_compute),
                "unused_compute": int(winner.unused_compute),
                "verifier_calls": int(winner.verifier_calls),
                "verifier_candidate_scores": int(winner.verifier_candidate_scores),
                "schedule_notation": winner.notation,
                "lcb80": float(winner.lcb80),
                "predicted_delta": float(winner.predicted_delta),
                "empirical_delta": float(winner.empirical_delta),
                "ci95_low": float(winner.ci95_low),
                "ci95_high": float(winner.ci95_high),
                "model": "runwayml/stable-diffusion-v1-5",
                "steps": STEPS,
                "eta": 0.0,
                "guidance_scale": 7.5,
                "calibration_prompt_corpus": "ImageReward test_ir.json",
                "calibration_prompt_count": BANK_PROMPT_COUNT,
                "selection_seed": 20260920,
                "fold_seed": 20260920,
                "subset_seed": SUBSET_SEED,
                "subsets_per_prompt": SUBSETS_PER_PROMPT,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "lcb_quantile": LCB_QUANTILE,
                "candidate_seed_base": 20260920,
                "selection_rule": "argmax over policies of the 80% lower confidence bound of bootstrap predicted Delta IR",
                "oof_gate": gate,
                "prompt_manifest_sha256": sha256(MANIFEST),
                "cell_estimates_sha256": sha256(REPLAY / "cell_estimates.csv"),
            }
            (EXP / "FROZEN_SCHEDULE.json").write_text(json.dumps(frozen, indent=2) + "\n")
    overlap = manifest["exact_overlap_with_previous_120_prompt_calibration"]
    overlap_note = f"overlap with the previous 120-prompt calibration = {overlap}."
    write_report(policies, stats, full_agg, cells, oof_frame, gate, bootstrap, frozen, overlap_note)
    summary = {
        "invariant_max_abs_error": invariant,
        "oof_gate": gate,
        "frozen": frozen,
        "top5_lcb": bootstrap.head(5).to_dict("records") if not bootstrap.empty else [],
        "psp_mean_ir": full_agg["q_psp"],
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
