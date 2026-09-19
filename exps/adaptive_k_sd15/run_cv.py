#!/usr/bin/env python3
"""Prompt-grouped OOF calibration and gate for Adaptive-K.

For every fold: fit the regret predictors on 160 prompts, evaluate on 40 unseen
prompts, and simulate the exact aggregate-budget Adaptive-K policy against PSP,
fixed 10->1, fixed 10->2 and fixed 10->3.  No GenEval validation is launched here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from regret_models import (
    FEATURES,
    BoostedModelB,
    IsotonicModelA,
    exact_budget_allocation,
    project_monotone,
)

EXP = Path(__file__).resolve().parent
CAL = EXP / "calibration"
DATASET = CAL / "subset_dataset.parquet"
SUBSETS_PER_PROMPT = 1000
N_FOLDS = 5
GATE_MIN_POSITIVE_FIXED_FOLDS = 4
PREFERRED_MIN_POSITIVE_PSP_FOLDS = 3


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    value = spearmanr(a, b).statistic
    return float(value)


def regression_metrics(predicted: np.ndarray, realized: np.ndarray) -> dict:
    rows = {}
    for k in range(3):
        error = predicted[:, k] - realized[:, k]
        rows[f"mae_K{k + 1}"] = float(np.mean(np.abs(error)))
        rows[f"rmse_K{k + 1}"] = float(math.sqrt(float(np.mean(error**2))))
        rows[f"spearman_K{k + 1}"] = safe_spearman(predicted[:, k], realized[:, k])
    rows["mae_b12"] = float(np.mean(np.abs((predicted[:, 0] - predicted[:, 1]) - (realized[:, 0] - realized[:, 1]))))
    rows["mae_b23"] = float(np.mean(np.abs((predicted[:, 1] - predicted[:, 2]) - (realized[:, 1] - realized[:, 2]))))
    rows["spearman_b12"] = safe_spearman(predicted[:, 0] - predicted[:, 1], realized[:, 0] - realized[:, 1])
    rows["spearman_b23"] = safe_spearman(predicted[:, 1] - predicted[:, 2], realized[:, 1] - realized[:, 2])
    return rows


def policy_simulation(predicted: np.ndarray, topk: np.ndarray, psp: np.ndarray, regret: np.ndarray) -> dict:
    """predicted (P,S,3), topk (P,S,3), psp (P,S), regret (P,S,3)."""
    prompts, subsets, _ = predicted.shape
    adaptive_sum = psp_sum = fixed1_sum = fixed2_sum = fixed3_sum = 0.0
    counts = np.zeros(4, dtype=int)
    adaptive_per_prompt = np.zeros((prompts, subsets))
    chosen_k_sum = np.zeros(prompts)
    chosen_regret_sum = np.zeros(prompts)
    fixed2_per_prompt = topk[:, :, 1]
    for b in range(subsets):
        chosen = exact_budget_allocation(predicted[:, b, :])
        adaptive = topk[np.arange(prompts), b, chosen - 1]
        adaptive_per_prompt[:, b] = adaptive
        chosen_k_sum += chosen
        chosen_regret_sum += regret[np.arange(prompts), b, chosen - 1]
        adaptive_sum += float(adaptive.mean())
        psp_sum += float(psp[:, b].mean())
        fixed1_sum += float(topk[:, b, 0].mean())
        fixed2_sum += float(topk[:, b, 1].mean())
        fixed3_sum += float(topk[:, b, 2].mean())
        counts += np.bincount(chosen, minlength=4)
    draws = subsets
    assert counts[1] == counts[3], (counts[1], counts[3])
    return {
        "adaptive_ir": adaptive_sum / draws,
        "psp_ir": psp_sum / draws,
        "fixed1_ir": fixed1_sum / draws,
        "fixed2_ir": fixed2_sum / draws,
        "fixed3_ir": fixed3_sum / draws,
        "delta_vs_psp": (adaptive_sum - psp_sum) / draws,
        "delta_vs_fixed1": (adaptive_sum - fixed1_sum) / draws,
        "delta_vs_fixed2": (adaptive_sum - fixed2_sum) / draws,
        "delta_vs_fixed3": (adaptive_sum - fixed3_sum) / draws,
        "k_mean": float((counts[1] + 2 * counts[2] + 3 * counts[3]) / (draws * prompts)),
        "k1_fraction": float(counts[1] / (draws * prompts)),
        "k2_fraction": float(counts[2] / (draws * prompts)),
        "k3_fraction": float(counts[3] / (draws * prompts)),
        "adaptive_per_prompt": adaptive_per_prompt.mean(axis=1),
        "fixed2_per_prompt": fixed2_per_prompt.mean(axis=1),
        "chosen_k_per_prompt": chosen_k_sum / draws,
        "chosen_regret_per_prompt": chosen_regret_sum / draws,
        "prompt_index": np.arange(prompts),
    }


def main() -> None:
    frame = pd.read_parquet(DATASET)
    features_all = frame[list(FEATURES)].to_numpy()
    labels_all = frame[["L1", "L2", "L3"]].to_numpy()
    folds = sorted(frame.fold.unique().tolist())
    assert folds == list(range(N_FOLDS))

    per_fold = []
    metrics_rows = []
    oof_frames = []
    importance = {}
    for fold in folds:
        train = frame.fold.ne(fold).to_numpy()
        test = ~train
        model_a = IsotonicModelA().fit(features_all[train], labels_all[train])
        model_b = BoostedModelB().fit(features_all[train], labels_all[train])
        pred_a = project_monotone(model_a.predict(features_all[test]))
        pred_b_raw = model_b.predict(features_all[test])
        pred_b = project_monotone(pred_b_raw)
        realized = labels_all[test]
        metrics_rows.append({
            "fold": fold,
            "model": "A_isotonic_projected",
            **regression_metrics(pred_a, realized),
        })
        metrics_rows.append({
            "fold": fold,
            "model": "B_boosted_unprojected",
            **regression_metrics(pred_b_raw, realized),
        })
        metrics_rows.append({
            "fold": fold,
            "model": "B_boosted_projected",
            **regression_metrics(pred_b, realized),
        })
        importance[fold] = model_b.feature_importance().tolist()

        test_frame = frame[test].sort_values(["prompt_id", "subset_index"]).reset_index(drop=True)
        prompt_ids = sorted(test_frame.prompt_id.unique().tolist())
        prompts = len(prompt_ids)
        topk = test_frame[["topk1", "topk2", "topk3"]].to_numpy().reshape(prompts, SUBSETS_PER_PROMPT, 3)
        psp = test_frame["psp_final"].to_numpy().reshape(prompts, SUBSETS_PER_PROMPT)
        regret = test_frame[["L1", "L2", "L3"]].to_numpy().reshape(prompts, SUBSETS_PER_PROMPT, 3)
        pred_primary = pred_b.reshape(prompts, SUBSETS_PER_PROMPT, 3)
        pred_simple = pred_a.reshape(prompts, SUBSETS_PER_PROMPT, 3)
        sim_primary = policy_simulation(pred_primary, topk, psp, regret)
        sim_simple = policy_simulation(pred_simple, topk, psp, regret)
        per_fold.append({
            "fold": fold,
            "n_prompts": prompts,
            "prompt_ids": prompt_ids,
            "primary": sim_primary,
            "secondary_isotonic": sim_simple,
        })
        oof_frames.append(pd.DataFrame({
            "fold": fold,
            "prompt_id": test_frame.prompt_id.to_numpy(),
            "subset_index": test_frame.subset_index.to_numpy(),
            "L1": realized[:, 0],
            "L2": realized[:, 1],
            "L3": realized[:, 2],
            "pred_primary_L1": pred_b[:, 0],
            "pred_primary_L2": pred_b[:, 1],
            "pred_primary_L3": pred_b[:, 2],
            "pred_isotonic_L1": pred_a[:, 0],
            "pred_isotonic_L2": pred_a[:, 1],
            "pred_isotonic_L3": pred_a[:, 2],
        }))
        print(json.dumps({"fold": fold, "primary_delta_vs_psp": sim_primary["delta_vs_psp"],
                          "primary_delta_vs_fixed2": sim_primary["delta_vs_fixed2"],
                          "k1": sim_primary["k1_fraction"], "k3": sim_primary["k3_fraction"]}))

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(CAL / "regret_metrics.csv", index=False)
    pd.concat(oof_frames, ignore_index=True).to_parquet(CAL / "oof_predictions.parquet", index=False)

    fold_table = pd.DataFrame([
        {
            "fold": row["fold"],
            "psp_ir": row["primary"]["psp_ir"],
            "fixed1_ir": row["primary"]["fixed1_ir"],
            "fixed2_ir": row["primary"]["fixed2_ir"],
            "fixed3_ir": row["primary"]["fixed3_ir"],
            "adaptive_ir": row["primary"]["adaptive_ir"],
            "delta_vs_psp": row["primary"]["delta_vs_psp"],
            "delta_vs_fixed2": row["primary"]["delta_vs_fixed2"],
            "k1_fraction": row["primary"]["k1_fraction"],
            "k2_fraction": row["primary"]["k2_fraction"],
            "k3_fraction": row["primary"]["k3_fraction"],
        }
        for row in per_fold
    ])
    fold_table.to_csv(CAL / "cv_results.csv", index=False)
    secondary = pd.DataFrame([{
        "fold": row["fold"],
        "adaptive_ir": row["secondary_isotonic"]["adaptive_ir"],
        "delta_vs_psp": row["secondary_isotonic"]["delta_vs_psp"],
        "delta_vs_fixed2": row["secondary_isotonic"]["delta_vs_fixed2"],
    } for row in per_fold])
    secondary.to_csv(CAL / "cv_results_isotonic.csv", index=False)

    mean_delta_psp = float(fold_table.delta_vs_psp.mean())
    mean_delta_fixed2 = float(fold_table.delta_vs_fixed2.mean())
    positive_fixed_folds = int((fold_table.delta_vs_fixed2 >= 0).sum())
    positive_psp_folds = int((fold_table.delta_vs_psp >= 0).sum())
    gate = {
        "mean_delta_vs_psp": mean_delta_psp,
        "condition_A_mean_delta_vs_psp_positive": mean_delta_psp > 0,
        "mean_delta_vs_fixed10to2": mean_delta_fixed2,
        "condition_B_mean_delta_vs_fixed10to2_positive": mean_delta_fixed2 > 0,
        "positive_folds_vs_fixed10to2": positive_fixed_folds,
        "condition_C_at_least_4_of_5_fixed_folds_nonnegative": positive_fixed_folds >= GATE_MIN_POSITIVE_FIXED_FOLDS,
        "positive_folds_vs_psp": positive_psp_folds,
        "preferred_at_least_3_of_5_psp_folds_nonnegative": positive_psp_folds >= PREFERRED_MIN_POSITIVE_PSP_FOLDS,
    }
    gate["pass"] = bool(
        gate["condition_A_mean_delta_vs_psp_positive"]
        and gate["condition_B_mean_delta_vs_fixed10to2_positive"]
        and gate["condition_C_at_least_4_of_5_fixed_folds_nonnegative"]
    )
    (CAL / "oof_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    aggregated_importance = np.mean([np.array(v) for v in importance.values()], axis=0)
    importance_frame = pd.DataFrame({"feature": list(FEATURES), "importance": aggregated_importance}).sort_values(
        "importance", ascending=False
    )
    importance_frame.to_csv(CAL / "feature_importance.csv", index=False)

    make_plots(frame, oof_frames, per_fold, fold_table)

    frozen = None
    if gate["pass"]:
        final = BoostedModelB().fit(features_all, labels_all)
        policy_dir = EXP / "policy"
        policy_dir.mkdir(exist_ok=True)
        import joblib

        joblib.dump(final, policy_dir / "adaptive_k_regret_model.joblib")
        (policy_dir / "feature_config.json").write_text(json.dumps({
            "features": list(FEATURES),
            "normalization": "median / MAD with eps=1e-6",
        }, indent=2) + "\n")
        frozen = {
            "M": 10,
            "t": 16,
            "K_choices": [1, 2, 3],
            "budget_rule": "sum_p K_p == 2N",
            "steps": 64,
            "eta": 0.0,
            "guidance_scale": 7.5,
            "model": "runwayml/stable-diffusion-v1-5",
            "primary_predictor": "Model B (GradientBoostingRegressor, 200 trees, depth 3) + monotone projection",
            "gate": gate,
            "feature_importance_top": importance_frame.head(6).to_dict("records"),
        }
        (policy_dir / "policy_config.json").write_text(json.dumps(frozen, indent=2) + "\n")

    write_report(frame, metrics, fold_table, secondary, gate, importance_frame, frozen)
    print(json.dumps({"gate": gate, "frozen": frozen}, indent=2, default=str))


def make_plots(frame, oof_frames, per_fold, fold_table) -> None:
    plots = CAL / "plots"
    plots.mkdir(exist_ok=True)
    oof = pd.concat(oof_frames, ignore_index=True)
    sample = oof.sample(n=min(20000, len(oof)), random_state=0)
    for k in range(3):
        plt.figure(figsize=(4, 4))
        plt.scatter(sample[f"pred_primary_L{k+1}"], sample[f"L{k+1}"], s=2, alpha=0.15)
        limit = max(sample[f"pred_primary_L{k+1}"].max(), sample[f"L{k+1}"].max())
        plt.plot([0, limit], [0, limit], "r--", linewidth=1)
        plt.xlabel(f"predicted L{k+1}")
        plt.ylabel(f"realized L{k+1}")
        plt.title(f"Regret K={k+1}")
        plt.tight_layout()
        plt.savefig(plots / f"A_regret_K{k+1}.png", dpi=120)
        plt.close()
    # B: mean boundary gap g2 vs mean chosen K per held-out prompt.
    g2_values = []
    k_values = []
    regret_values = []
    for row in per_fold:
        pids = row["prompt_ids"]
        mean_g2 = frame[frame.prompt_id.isin(pids)].groupby("prompt_id").g2.mean().reindex(pids).to_numpy()
        g2_values.append(mean_g2)
        k_values.append(row["primary"]["chosen_k_per_prompt"])
        regret_values.append(row["primary"]["chosen_regret_per_prompt"])
    g2_all = np.concatenate(g2_values)
    k_all = np.concatenate(k_values)
    regret_all = np.concatenate(regret_values)
    plt.figure(figsize=(5, 3))
    plt.scatter(g2_all, k_all, s=8, alpha=0.5)
    plt.xlabel("mean g2 = z2 - z3 at step 16")
    plt.ylabel("mean chosen K")
    plt.title("Boundary gap vs allocated K")
    plt.tight_layout()
    plt.savefig(plots / "B_boundary_gap_g2.png", dpi=120)
    plt.close()
    # C/D: predicted marginal benefits.
    for name, values in (
        ("C_predicted_b12", oof.pred_primary_L1 - oof.pred_primary_L2),
        ("D_predicted_b23", oof.pred_primary_L2 - oof.pred_primary_L3),
    ):
        plt.figure(figsize=(5, 3))
        plt.hist(values, bins=80)
        plt.xlabel(name.replace("_", " "))
        plt.ylabel("subset count")
        plt.title(name.replace("_", " "))
        plt.tight_layout()
        plt.savefig(plots / f"{name}.png", dpi=120)
        plt.close()
    # E: realized regret at the allocated K, grouped by allocated K.
    plt.figure(figsize=(5, 3))
    for k in (1, 2, 3):
        mask = np.abs(k_all - k) < 0.25
        if mask.any():
            plt.bar(str(k), regret_all[mask].mean())
    plt.xlabel("allocated K")
    plt.ylabel("mean realized regret")
    plt.title("Realized regret by allocated K")
    plt.tight_layout()
    plt.savefig(plots / "E_regret_by_allocated_K.png", dpi=120)
    plt.close()
    for name, label in (("delta_vs_psp", "Adaptive - PSP"), ("delta_vs_fixed2", "Adaptive - fixed10->2")):
        plt.figure(figsize=(5, 3))
        plt.bar(fold_table.fold.astype(str), fold_table[name])
        plt.axhline(0, color="k", linewidth=0.8)
        plt.xlabel("fold")
        plt.ylabel("mean Delta IR")
        plt.title(label)
        plt.tight_layout()
        plt.savefig(plots / f"F_{name}.png", dpi=120)
        plt.close()
    plt.figure(figsize=(5, 3))
    plt.bar(fold_table.fold.astype(str), fold_table.k1_fraction, label="K=1")
    plt.bar(fold_table.fold.astype(str), fold_table.k2_fraction, bottom=fold_table.k1_fraction, label="K=2")
    plt.bar(
        fold_table.fold.astype(str),
        fold_table.k3_fraction,
        bottom=fold_table.k1_fraction + fold_table.k2_fraction,
        label="K=3",
    )
    plt.xlabel("fold")
    plt.ylabel("fraction")
    plt.title("Chosen-K distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "E_k_distribution.png", dpi=120)
    plt.close()


def write_report(frame, metrics, fold_table, secondary, gate, importance_frame, frozen) -> None:
    def metric_row(model_prefix, label):
        subset = metrics[metrics.model.eq(model_prefix)]
        return (
            f"| {label} | {subset.mae_K1.mean():.4f} | {subset.mae_K2.mean():.4f} | {subset.mae_K3.mean():.4f} | "
            f"{subset.rmse_K1.mean():.4f} | {subset.rmse_K2.mean():.4f} | {subset.rmse_K3.mean():.4f} | "
            f"{subset.spearman_K1.mean():.3f} | {subset.spearman_K2.mean():.3f} | {subset.spearman_K3.mean():.3f} |"
        )

    fold_rows = "\n".join(
        f"| {row.fold} | {row.psp_ir:.6f} | {row.fixed2_ir:.6f} | {row.adaptive_ir:.6f} | "
        f"{row.delta_vs_psp:+.6f} | {row.delta_vs_fixed2:+.6f} | {row.k1_fraction:.3f}/{row.k2_fraction:.3f}/{row.k3_fraction:.3f} |"
        for row in fold_table.itertuples()
    )
    importance_rows = "\n".join(
        f"| {row.feature} | {row.importance:.4f} |" for row in importance_frame.head(10).itertuples()
    )
    frozen_text = (
        f"**OOF gate PASSED.** The final policy predictor was trained on all 200 prompts and saved under "
        f"`policy/`. Frozen configuration: M=10, t=16, K in {{1,2,3}}, exact aggregate budget sum K_p = 2N."
        if frozen
        else "**OOF gate FAILED.** Per the pre-registered stop rule, no predictor was frozen and no GenEval "
        "validation was launched. Failure is reported, not tuned away."
    )
    report = f"""# Adaptive-K calibration report

## Outcome

{frozen_text}

Method studied: 10 initial seeds -> step 16 -> observe the 10 step-16 ImageReward
scores -> choose K in {{1,2,3}} -> finish K candidates -> final winner by final
ImageReward.  Cost is `160 + 48K` per prompt, and the exact-budget policy enforces
`sum_p K_p = 2N`, giving exactly 256 average logical UNet evaluations per prompt.

## Required calibration table

| Fold | PSP IR | Fixed 10->2 IR | Adaptive-K IR | Delta Adaptive-PSP | Delta Adaptive-Fixed | K=1/2/3 |
|---|---:|---:|---:|---:|---:|---|
{fold_rows}

Mean Delta Adaptive-PSP = {gate['mean_delta_vs_psp']:+.6f}; mean Delta
Adaptive-fixed10->2 = {gate['mean_delta_vs_fixed10to2']:+.6f}.

## Regret-predictor metrics (held-out prompts)

| Model | MAE K=1 | MAE K=2 | MAE K=3 | RMSE K=1 | RMSE K=2 | RMSE K=3 | Spearman K=1 | Spearman K=2 | Spearman K=3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{metric_row('A_isotonic_projected', 'Model A isotonic (projected)')}
{metric_row('B_boosted_unprojected', 'Model B boosted (unprojected)')}
{metric_row('B_boosted_projected', 'Model B boosted (projected)')}

Monotone projection: Model B MAE K=1/2/3 changes from
{metrics[metrics.model.eq('B_boosted_unprojected')].mae_K1.mean():.4f}/{metrics[metrics.model.eq('B_boosted_unprojected')].mae_K2.mean():.4f}/{metrics[metrics.model.eq('B_boosted_unprojected')].mae_K3.mean():.4f}
to
{metrics[metrics.model.eq('B_boosted_projected')].mae_K1.mean():.4f}/{metrics[metrics.model.eq('B_boosted_projected')].mae_K2.mean():.4f}/{metrics[metrics.model.eq('B_boosted_projected')].mae_K3.mean():.4f}.

## Marginal-benefit metrics (held-out)

| Model | MAE b12 | MAE b23 | Spearman b12 | Spearman b23 |
|---|---:|---:|---:|---:|
| Model B projected | {metrics[metrics.model.eq('B_boosted_projected')].mae_b12.mean():.4f} | {metrics[metrics.model.eq('B_boosted_projected')].mae_b23.mean():.4f} | {metrics[metrics.model.eq('B_boosted_projected')].spearman_b12.mean():.3f} | {metrics[metrics.model.eq('B_boosted_projected')].spearman_b23.mean():.3f} |
| Model A isotonic | {metrics[metrics.model.eq('A_isotonic_projected')].mae_b12.mean():.4f} | {metrics[metrics.model.eq('A_isotonic_projected')].mae_b23.mean():.4f} | {metrics[metrics.model.eq('A_isotonic_projected')].spearman_b12.mean():.3f} | {metrics[metrics.model.eq('A_isotonic_projected')].spearman_b23.mean():.3f} |

## OOF gate

| Condition | Value | Pass |
|---|---:|---|
| A: mean Delta Adaptive-PSP > 0 | {gate['mean_delta_vs_psp']:+.6f} | {gate['condition_A_mean_delta_vs_psp_positive']} |
| B: mean Delta Adaptive-fixed10->2 > 0 | {gate['mean_delta_vs_fixed10to2']:+.6f} | {gate['condition_B_mean_delta_vs_fixed10to2_positive']} |
| C: >=4/5 folds Delta Adaptive-fixed10->2 >= 0 | {gate['positive_folds_vs_fixed10to2']}/5 | {gate['condition_C_at_least_4_of_5_fixed_folds_nonnegative']} |
| Preferred: >=3/5 folds positive vs PSP | {gate['positive_folds_vs_psp']}/5 | {gate['preferred_at_least_3_of_5_psp_folds_nonnegative']} |
| **Overall gate** |  | **{gate['pass']}** |

Secondary (Model A isotonic) mean Delta vs PSP = {secondary.delta_vs_psp.mean():+.6f},
vs fixed10->2 = {secondary.delta_vs_fixed2.mean():+.6f}.

## Chosen-K accounting

Each fold's exact-budget allocation satisfies `n1 + 2 n2 + 3 n3 = 2N` and hence
`n1 = n3` (verified per batch draw in `policy_simulation`).  Mean K per prompt is
2.0 by construction.

## Feature importance (Model B, averaged over folds)

| Feature | Importance |
|---|---:|
{importance_rows}

## Interpretation guardrails

This is calibration evidence only.  HPS and official GenEval were not inspected.
If the gate failed, no online GenEval validation was run and no parameter was
tuned in response.  A new policy requires a new development cycle.
"""
    (CAL / "CALIBRATION_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
