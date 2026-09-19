#!/usr/bin/env python3
"""Predictor v2: weighted policy classification, grouped OOF, headroom capture.

Feature families F0 (scalar step-16) and F1 (temporal reward dynamics) are
evaluated.  F2/F3 (latent drift / ImageReward hidden features) require a dynamics
bank and are added by a separate script.
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

from voi_v2_models import (
    F0_FEATURES,
    F1_EXTRA,
    WeightedBoost,
    WeightedLogistic,
    best_threshold,
    cap_weights,
    oracle_headroom_capture,
)

EXP = Path(__file__).resolve().parent
CAL = EXP / "calibration"
POOLS = CAL / "pool_examples_f01.parquet"
N_FOLDS = 5
MIN_CAPTURE = 0.65
STRONG_CAPTURE = 0.70
FEATURE_SETS = {
    "F0": list(F0_FEATURES),
    "F1": list(F0_FEATURES) + list(F1_EXTRA),
}


def prompt_aggregate(values: np.ndarray, prompt_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(prompt_ids, kind="stable")
    sorted_prompts = prompt_ids[order]
    sorted_values = values[order]
    unique, counts = np.unique(sorted_prompts, return_counts=True)
    sums = np.add.reduceat(sorted_values, np.concatenate([[0], np.cumsum(counts)[:-1]]))
    return unique, sums / counts


def safe_spearman(a, b):
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def main() -> None:
    frame = pd.read_parquet(POOLS)
    prompt_ids = frame.prompt_id.to_numpy()
    folds = sorted(frame.fold.unique().tolist())
    y_a = frame.Y_A.to_numpy()
    y_b = frame.Y_B.to_numpy()
    y_psp = frame.Y_PSP.to_numpy()
    y_oracle = frame.oracle.to_numpy()
    v = frame.V.to_numpy()
    z = (v > 0).astype(int)
    _, psp_prompt = prompt_aggregate(y_psp, prompt_ids)
    _, a_prompt = prompt_aggregate(y_a, prompt_ids)
    _, b_prompt = prompt_aggregate(y_b, prompt_ids)
    _, oracle_prompt = prompt_aggregate(y_oracle, prompt_ids)
    q_a, q_b, q_psp, q_oracle = (
        float(a_prompt.mean()), float(b_prompt.mean()), float(psp_prompt.mean()), float(oracle_prompt.mean())
    )
    baseline_table = {"Q_A": q_a, "Q_B": q_b, "Q_PSP": q_psp, "Q_oracle": q_oracle}
    print(json.dumps({"baseline": baseline_table}, indent=2))

    fold_rows = []
    family_rows = []
    oof_probs = {}
    oof_picked = {}
    for family, columns in FEATURE_SETS.items():
        features = frame[columns].to_numpy()
        for model_name, factory in (("logistic", WeightedLogistic), ("boost", WeightedBoost)):
            probs = np.full(len(frame), np.nan)
            deltas = []
            for fold in folds:
                train = frame.fold.ne(fold).to_numpy()
                test = ~train
                weight = cap_weights(np.abs(v[train]))
                model = factory().fit(features[train], z[train], weight)
                train_prob = model.predict_proba(features[train])
                delta = best_threshold(train_prob, y_a[train], y_b[train])
                deltas.append(delta)
                probs[test] = model.predict_proba(features[test])
            key = f"{family}:{model_name}"
            oof_probs[key] = probs
            # Apply each fold's frozen threshold to that fold only.
            picked = np.zeros(len(frame), dtype=bool)
            for fold, delta in zip(folds, deltas):
                mask = frame.fold.eq(fold).to_numpy()
                picked[mask] = probs[mask] > delta
            oof_picked[key] = picked
            y_policy = np.where(picked, y_b, y_a)
            unique, policy_prompt = prompt_aggregate(y_policy, prompt_ids)
            per_prompt = pd.DataFrame({
                "prompt_id": unique,
                "policy": policy_prompt,
                "Y_A": a_prompt,
                "Y_B": b_prompt,
                "Y_PSP": psp_prompt,
                "oracle": oracle_prompt,
            }).merge(frame[["prompt_id", "fold"]].drop_duplicates(), on="prompt_id")
            fold_record = []
            for fold, group in per_prompt.groupby("fold"):
                fold_record.append({
                    "family": family, "model": f"{family}:{model_name}", "fold": int(fold),
                    "policy": group.policy.mean(), "Y_A": group.Y_A.mean(), "Y_B": group.Y_B.mean(),
                    "Y_PSP": group.Y_PSP.mean(), "oracle": group.oracle.mean(),
                    "delta_vs_psp": (group.policy - group.Y_PSP).mean(),
                    "delta_vs_A": (group.policy - group.Y_A).mean(),
                    "delta_vs_B": (group.policy - group.Y_B).mean(),
                })
            fold_df = pd.DataFrame(fold_record)
            fold_rows.append(fold_df)
            q_policy = float(per_prompt.policy.mean())
            family_rows.append({
                "family": family,
                "model": f"{family}:{model_name}",
                "q_policy": q_policy,
                "delta_vs_A": float((per_prompt.policy - per_prompt.Y_A).mean()),
                "delta_vs_B": float((per_prompt.policy - per_prompt.Y_B).mean()),
                "delta_vs_psp": float((per_prompt.policy - per_prompt.Y_PSP).mean()),
                "capture": oracle_headroom_capture(q_policy, q_a, q_b, q_oracle),
                "positive_psp_folds": int((fold_df.delta_vs_psp >= 0).sum()),
                "auc": float(roc_auc(z, probs)),
                "spearman_p_V": safe_spearman(probs, v),
                "pct_choose_B": float(picked.mean()),
                "delta_A": deltas,
            })
    families = pd.DataFrame(family_rows)
    families.to_csv(CAL / "family_results.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(CAL / "fold_results.csv", index=False)

    # Per-family primary model = highest OOF policy value.
    primary = families.sort_values("q_policy", ascending=False).groupby("family", as_index=False).first()
    primary.to_csv(CAL / "family_primary.csv", index=False)
    best_overall = families.sort_values("q_policy", ascending=False).iloc[0]
    overall_model = best_overall.model
    probs = oof_probs[overall_model]

    gate = {
        "best_family_model": overall_model,
        "q_policy": float(best_overall.q_policy),
        "delta_vs_psp": float(best_overall.delta_vs_psp),
        "capture": float(best_overall.capture),
        "positive_psp_folds": int(best_overall.positive_psp_folds),
        "condition_beats_psp": bool(best_overall.delta_vs_psp > 0),
        "condition_4_of_5_folds": bool(best_overall.positive_psp_folds >= 4),
        "condition_capture_gt_65pct": bool(best_overall.capture > MIN_CAPTURE),
        "strong_capture_ge_70pct": bool(best_overall.capture >= STRONG_CAPTURE),
        "baseline": baseline_table,
    }
    gate["pass"] = bool(
        gate["condition_beats_psp"] and gate["condition_4_of_5_folds"] and gate["condition_capture_gt_65pct"]
    )
    (CAL / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    diagnostics = high_value_diagnostics(v, oof_picked[overall_model])
    diagnostics.to_csv(CAL / "high_value_diagnostics.csv", index=False)
    make_plots(probs, v, families)
    write_report(baseline_table, families, primary, gate, diagnostics)

    date = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    (CAL / "COMPLETE_F01").write_text(date + "\n")
    print(json.dumps({"gate": gate, "primary": primary.to_dict("records")}, indent=2))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def high_value_diagnostics(v, selected) -> pd.DataFrame:
    positives = v > 0
    order = np.argsort(-v)
    rows = []
    positive_total = float(v[positives].sum())
    captured = float(v[positives & selected].sum())
    negative_mass = float(-v[(v < 0) & selected].sum())
    for fraction, label in ((0.10, "top10pct"), (0.25, "top25pct"), (0.50, "top50pct")):
        count = max(1, int(round(positives.sum() * fraction)))
        target = np.zeros(len(v), dtype=bool)
        target[order[:count]] = True
        tp = int((target & selected).sum())
        rows.append({
            "group": label,
            "positive_pools": count,
            "caught": tp,
            "precision": tp / max(1, int(selected.sum())),
            "recall": tp / count,
        })
    rows.append({
        "group": "mass",
        "captured_positive_mass": captured,
        "positive_mass": positive_total,
        "captured_fraction": captured / positive_total if positive_total > 0 else float("nan"),
        "incurred_negative_mass": negative_mass,
        "net_mass": captured - negative_mass,
    })
    return pd.DataFrame(rows)


def make_plots(probs, v, families) -> None:
    plots = CAL / "plots"
    plots.mkdir(exist_ok=True)
    sample = np.random.default_rng(0).integers(0, len(v), size=min(20000, len(v)))
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(probs[sample], v[sample], s=2, alpha=0.15)
    plt.axvline(0.5, color="k", linewidth=0.8)
    plt.axhline(0.0, color="k", linewidth=0.8)
    plt.xlabel("predicted P(B beneficial)")
    plt.ylabel("realized V")
    plt.title("weighted classifier OOF")
    plt.tight_layout()
    plt.savefig(plots / "probability_vs_V.png", dpi=120)
    plt.close()
    plt.figure(figsize=(6, 3))
    labels = families.model.tolist()
    plt.bar(range(len(labels)), families.capture.values)
    plt.axhline(0.62, color="r", linestyle="--", linewidth=1, label="62% PSP threshold")
    plt.axhline(1.0, color="g", linestyle="--", linewidth=1, label="oracle")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.ylabel("oracle headroom captured")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "headroom_capture.png", dpi=120)
    plt.close()


def write_report(baseline, families, primary, gate, diagnostics) -> None:
    family_rows = "\n".join(
        f"| {row.family} | {row.model.split(':')[1]} | {row.q_policy:.6f} | {row.delta_vs_B:+.6f} | "
        f"{row.delta_vs_psp:+.6f} | {100 * row.capture:.1f}% | {row.positive_psp_folds}/5 | {row.auc:.3f} | {row.pct_choose_B:.3f} |"
        for row in families.sort_values("q_policy", ascending=False).itertuples()
    )
    diagnostic_rows = "\n".join(
        f"| {row.group} | {getattr(row, 'caught', '')} | {getattr(row, 'precision', '')} | {getattr(row, 'recall', '')} |"
        for row in diagnostics.itertuples()
    )
    head = (
        f"**Gate {'PASS' if gate['pass'] else 'FAIL'}** for the best family/model "
        f"`{gate['best_family_model']}`: Q_policy = {gate['q_policy']:.6f}, "
        f"Delta vs PSP = {gate['delta_vs_psp']:+.6f}, capture = {100 * gate['capture']:.1f}%, "
        f"positive PSP folds = {gate['positive_psp_folds']}/5."
    )
    report = f"""# Predictor v2 calibration report (F0 / F1)

## Outcome

{head}

Objective: weighted classification of `z = 1[V>0]` with weight `w = min(|V|, q95(|V|))`,
which is Bayes-aligned with maximising final ImageReward (choosing B iff
`E[V|X] > 0`).  Threshold is selected on training folds only.

## Baselines (prompt-level)

| Q_A | Q_B | Q_PSP | Q_oracle | Oracle-B | Oracle-PSP |
|---:|---:|---:|---:|---:|---:|
| {baseline['Q_A']:.6f} | {baseline['Q_B']:.6f} | {baseline['Q_PSP']:.6f} | {baseline['Q_oracle']:.6f} | {baseline['Q_oracle'] - baseline['Q_B']:+.6f} | {baseline['Q_oracle'] - baseline['Q_PSP']:+.6f} |

PSP requires capture >= 62% of the `Oracle - B` headroom
(`(Q_PSP - Q_B)/(Q_oracle - Q_B) = {(baseline['Q_PSP'] - baseline['Q_B']) / (baseline['Q_oracle'] - baseline['Q_B']):.3f}`).

## Families (all models; primary per family is the best OOF policy value)

| Family | Model | Q_policy | Delta vs B | Delta vs PSP | Headroom captured | PSP folds | AUC | % choose B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{family_rows}

## High-value positive diagnostics (OOF, best model)

| Group | caught | precision | recall |
|---|---:|---:|---:|
{diagnostic_rows}

## Gate

| Condition | Value | Pass |
|---|---:|---|
| best Delta vs PSP > 0 | {gate['delta_vs_psp']:+.6f} | {gate['condition_beats_psp']} |
| >= 4/5 folds non-negative vs PSP | {gate['positive_psp_folds']}/5 | {gate['condition_4_of_5_folds']} |
| capture > 65% | {100 * gate['capture']:.1f}% | {gate['condition_capture_gt_65pct']} |
| strong capture >= 70% | {100 * gate['capture']:.1f}% | {gate['strong_capture_ge_70pct']} |
| **Overall** |  | **{gate['pass']}** |

F2 (latent drift) and F3 (combined + ImageReward hidden features) require a
dynamics bank and are not part of this F0/F1 run.
"""
    (CAL / "CALIBRATION_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
