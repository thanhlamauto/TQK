#!/usr/bin/env python3
"""Predictor v2 OOF with all four feature families F0-F3.

F3 uses the frozen step-16 ImageReward hidden features, reduced with PCA(16) fit
on training folds only, then aggregated over each pool's Top-4 candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from run_cv_v2 import high_value_diagnostics, make_plots, prompt_aggregate, safe_spearman
from voi_v2_models import (
    F0_FEATURES,
    F1_EXTRA,
    F2_FEATURES,
    WeightedBoost,
    WeightedLogistic,
    best_threshold,
    cap_weights,
    oracle_headroom_capture,
)

EXP = Path(__file__).resolve().parent
CAL = EXP / "calibration"
POOLS = CAL / "pool_examples_dynamics.parquet"
HIDDEN = CAL / "hidden16.npz"
N_FOLDS = 5
PCA_DIM = 16
MIN_CAPTURE = 0.65
MODEL_TYPES = (("logistic", WeightedLogistic), ("boost", WeightedBoost))
F1_COLUMNS = list(F0_FEATURES) + list(F1_EXTRA)
F2_COLUMNS = F1_COLUMNS + list(F2_FEATURES)
HIDDEN_MEAN = [f"hmean_{i}" for i in range(PCA_DIM)]
HIDDEN_SPREAD = [f"hspread_{i}" for i in range(PCA_DIM)]
F3_COLUMNS = F2_COLUMNS + HIDDEN_MEAN + HIDDEN_SPREAD
FAMILIES = {
    "F0": list(F0_FEATURES),
    "F1": F1_COLUMNS,
    "F2": F2_COLUMNS,
    "F3": F3_COLUMNS,
}


def hidden_features(frame: pd.DataFrame, hidden: np.ndarray, prompt_index: dict, train_mask: np.ndarray) -> np.ndarray:
    train_prompts = frame.loc[train_mask, "prompt_id"].unique()
    train_rows = np.concatenate([hidden[prompt_index[p]] for p in train_prompts], axis=0)
    pca = PCA(n_components=PCA_DIM, random_state=20260920).fit(train_rows)
    transformed = pca.transform(hidden.reshape(-1, hidden.shape[-1])).reshape(hidden.shape[0], hidden.shape[1], PCA_DIM)
    row_prompt = frame.prompt_id.map(prompt_index).to_numpy()
    top4 = frame[[f"top4_{k}" for k in range(4)]].to_numpy()
    gathered = transformed[row_prompt[:, None], top4]  # (n,4,PCA_DIM)
    mean = gathered.mean(axis=1)
    spread = gathered[:, 2:4].mean(axis=1) - gathered[:, :2].mean(axis=1)
    return np.column_stack([mean, spread])


def main() -> None:
    frame = pd.read_parquet(POOLS)
    data = np.load(HIDDEN)
    hidden = data["hidden"].astype(np.float32)
    prompt_index = {int(p): i for i, p in enumerate(data["prompt_ids"])}
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
    baseline = {"Q_A": q_a, "Q_B": q_b, "Q_PSP": q_psp, "Q_oracle": q_oracle}
    print(json.dumps({"baseline": baseline}, indent=2))

    fold_rows = []
    family_rows = []
    oof_probs = {}
    oof_picked = {}
    for family, columns in FAMILIES.items():
        features = None if family == "F3" else frame[columns].to_numpy()
        for model_name, factory in MODEL_TYPES:
            probs = np.full(len(frame), np.nan)
            deltas = []
            for fold in folds:
                train = frame.fold.ne(fold).to_numpy()
                test = ~train
                if family == "F3":
                    fold_hidden = hidden_features(frame, hidden, prompt_index, train)
                    train_features = np.column_stack([frame.loc[train, F2_COLUMNS].to_numpy(), fold_hidden[train]])
                    test_features = np.column_stack([frame.loc[test, F2_COLUMNS].to_numpy(), fold_hidden[test]])
                else:
                    train_features, test_features = features[train], features[test]
                weight = cap_weights(np.abs(v[train]))
                model = factory().fit(train_features, z[train], weight)
                delta = best_threshold(model.predict_proba(train_features), y_a[train], y_b[train])
                deltas.append(delta)
                probs[test] = model.predict_proba(test_features)
            key = f"{family}:{model_name}"
            oof_probs[key] = probs
            picked = np.zeros(len(frame), dtype=bool)
            for fold, delta in zip(folds, deltas):
                mask = frame.fold.eq(fold).to_numpy()
                picked[mask] = probs[mask] > delta
            oof_picked[key] = picked
            y_policy = np.where(picked, y_b, y_a)
            unique, policy_prompt = prompt_aggregate(y_policy, prompt_ids)
            per_prompt = pd.DataFrame({
                "prompt_id": unique, "policy": policy_prompt, "Y_A": a_prompt,
                "Y_B": b_prompt, "Y_PSP": psp_prompt, "oracle": oracle_prompt,
            }).merge(frame[["prompt_id", "fold"]].drop_duplicates(), on="prompt_id")
            fold_record = []
            for fold, group in per_prompt.groupby("fold"):
                fold_record.append({
                    "family": family, "model": key, "fold": int(fold),
                    "policy": group.policy.mean(), "Y_A": group.Y_A.mean(), "Y_B": group.Y_B.mean(),
                    "Y_PSP": group.Y_PSP.mean(), "oracle": group.oracle.mean(),
                    "delta_vs_psp": (group.policy - group.Y_PSP).mean(),
                    "delta_vs_B": (group.policy - group.Y_B).mean(),
                })
            fold_df = pd.DataFrame(fold_record)
            fold_rows.append(fold_df)
            from sklearn.metrics import roc_auc_score
            q_policy = float(per_prompt.policy.mean())
            family_rows.append({
                "family": family, "model": key, "q_policy": q_policy,
                "delta_vs_B": float((per_prompt.policy - per_prompt.Y_B).mean()),
                "delta_vs_psp": float((per_prompt.policy - per_prompt.Y_PSP).mean()),
                "capture": oracle_headroom_capture(q_policy, q_a, q_b, q_oracle),
                "positive_psp_folds": int((fold_df.delta_vs_psp >= 0).sum()),
                "auc": float(roc_auc_score(z, probs)) if len(np.unique(z)) > 1 else float("nan"),
                "spearman_p_V": safe_spearman(probs, v),
                "pct_choose_B": float(picked.mean()),
            })
    families = pd.DataFrame(family_rows)
    families.to_csv(CAL / "family_results_dynamics.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(CAL / "fold_results_dynamics.csv", index=False)
    families.sort_values("q_policy", ascending=False).groupby("family", as_index=False).first().to_csv(
        CAL / "family_primary_dynamics.csv", index=False
    )
    best = families.sort_values("q_policy", ascending=False).iloc[0]
    probs = oof_probs[best.model]
    gate = {
        "best_family_model": best.model,
        "q_policy": float(best.q_policy),
        "delta_vs_psp": float(best.delta_vs_psp),
        "capture": float(best.capture),
        "positive_psp_folds": int(best.positive_psp_folds),
        "condition_beats_psp": bool(best.delta_vs_psp > 0),
        "condition_4_of_5_folds": bool(best.positive_psp_folds >= 4),
        "condition_capture_gt_65pct": bool(best.capture > MIN_CAPTURE),
        "strong_capture_ge_70pct": bool(best.capture >= 0.70),
        "baseline": baseline,
    }
    gate["pass"] = bool(
        gate["condition_beats_psp"] and gate["condition_4_of_5_folds"] and gate["condition_capture_gt_65pct"]
    )
    (CAL / "gate_dynamics.json").write_text(json.dumps(gate, indent=2) + "\n")
    diagnostics = high_value_diagnostics(v, oof_picked[best.model])
    diagnostics.to_csv(CAL / "high_value_diagnostics_dynamics.csv", index=False)
    make_plots(probs, v, families)
    write_dynamics_report(baseline, families, gate, diagnostics)
    print(json.dumps({"gate": gate}, indent=2))


def write_dynamics_report(baseline, families, gate, diagnostics) -> None:
    rows = "\n".join(
        f"| {r.family} | {r.model.split(':')[1]} | {r.q_policy:.6f} | {r.delta_vs_B:+.6f} | "
        f"{r.delta_vs_psp:+.6f} | {100 * r.capture:.1f}% | {r.positive_psp_folds}/5 | {r.auc:.3f} | {r.pct_choose_B:.3f} |"
        for r in families.sort_values(["family", "q_policy"], ascending=[True, False]).itertuples()
    )
    diag = "\n".join(
        f"| {r.group} | {getattr(r, 'caught', '')} | {getattr(r, 'precision', '')} | {getattr(r, 'recall', '')} |"
        for r in diagnostics.itertuples()
    )
    report = f"""# Predictor v2 calibration report (F0-F3)

## Outcome

Best family/model `{gate['best_family_model']}`: Q_policy = {gate['q_policy']:.6f},
Delta vs PSP = {gate['delta_vs_psp']:+.6f}, headroom capture = {100 * gate['capture']:.1f}%,
PSP folds = {gate['positive_psp_folds']}/5. Gate **{'PASS' if gate['pass'] else 'FAIL'}**.

Weighted classification of `z=1[V>0]` with weight `min(|V|, q95)`, threshold chosen
on training folds. F0 scalar step-16; F1 temporal reward dynamics (12/14/16);
F2 predicted-clean latent drift 14->16; F3 = F1+F2+ImageReward hidden PCA(16).

## Baselines (prompt-level)

| Q_A | Q_B | Q_PSP | Q_oracle |
|---:|---:|---:|---:|
| {baseline['Q_A']:.6f} | {baseline['Q_B']:.6f} | {baseline['Q_PSP']:.6f} | {baseline['Q_oracle']:.6f} |

PSP needs capture >= {(baseline['Q_PSP'] - baseline['Q_B']) / (baseline['Q_oracle'] - baseline['Q_B']):.3f} of the Oracle-B headroom.

## Families

| Family | Model | Q_policy | Delta vs B | Delta vs PSP | Headroom captured | PSP folds | AUC | % choose B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## High-value diagnostics (best model)

| Group | caught | precision | recall |
|---|---:|---:|---:|
{diag}

## Gate

| Condition | Value | Pass |
|---|---:|---|
| Delta vs PSP > 0 | {gate['delta_vs_psp']:+.6f} | {gate['condition_beats_psp']} |
| >= 4/5 folds non-negative | {gate['positive_psp_folds']}/5 | {gate['condition_4_of_5_folds']} |
| capture > 65% | {100 * gate['capture']:.1f}% | {gate['condition_capture_gt_65pct']} |
| **Overall** |  | **{gate['pass']}** |
"""
    (CAL / "CALIBRATION_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
