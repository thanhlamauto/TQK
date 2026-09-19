#!/usr/bin/env python3
"""Value-of-Information calibration: oracle headroom gate, then OOF selector.

The oracle gate is computed and reported BEFORE any predictor is trained.  If the
{A,B} action family has no oracle headroom over PSP, the script stops and reports
that the bottleneck is the action family, as required by the protocol.
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

from voi_models import (
    FEATURES,
    AmbiguityBaseline,
    BoostedModel,
    RidgeModel,
    choose_b,

    prompt_aggregate,
)

EXP = Path(__file__).resolve().parent
CAL = EXP / "calibration"
POOLS = CAL / "pool_examples.parquet"
N_FOLDS = 5
PRIMARY_MODEL = "boost"
PREFERRED_MIN_POSITIVE_PSP_FOLDS = 4


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def regression_metrics(predicted: np.ndarray, realized: np.ndarray) -> dict:
    error = predicted - realized
    positive = realized > 0
    chosen = predicted > 0
    tp = float(np.sum(positive & chosen))
    fp = float(np.sum(~positive & chosen))
    fn = float(np.sum(positive & ~chosen))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(float(np.mean(error**2)))),
        "spearman": safe_spearman(predicted, realized),
        "sign_accuracy": float(np.mean(positive == chosen)),
        "precision_V_pos": precision,
        "recall_V_pos": recall,
    }


def oracle_report(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    oracle = prompt_aggregate(frame.oracle.to_numpy(), frame.prompt_id.to_numpy())
    y_a = prompt_aggregate(frame.Y_A.to_numpy(), frame.prompt_id.to_numpy())
    y_b = prompt_aggregate(frame.Y_B.to_numpy(), frame.prompt_id.to_numpy())
    y_psp = prompt_aggregate(frame.Y_PSP.to_numpy(), frame.prompt_id.to_numpy())
    per_prompt = pd.DataFrame({
        "prompt_id": oracle["prompt_ids"],
        "oracle": oracle["per_prompt"],
        "Y_A": y_a["per_prompt"],
        "Y_B": y_b["per_prompt"],
        "Y_PSP": y_psp["per_prompt"],
    })
    per_prompt = per_prompt.merge(
        frame[["prompt_id", "fold"]].drop_duplicates(), on="prompt_id", how="left"
    )
    rows = []
    for fold, group in per_prompt.groupby("fold"):
        rows.append({
            "fold": int(fold),
            "n_prompts": len(group),
            "oracle": group.oracle.mean(),
            "Y_A": group.Y_A.mean(),
            "Y_B": group.Y_B.mean(),
            "Y_PSP": group.Y_PSP.mean(),
            "oracle_minus_psp": (group.oracle - group.Y_PSP).mean(),
            "oracle_minus_A": (group.oracle - group.Y_A).mean(),
            "oracle_minus_B": (group.oracle - group.Y_B).mean(),
        })
    rows.append({
        "fold": "overall",
        "n_prompts": len(per_prompt),
        "oracle": per_prompt.oracle.mean(),
        "Y_A": per_prompt.Y_A.mean(),
        "Y_B": per_prompt.Y_B.mean(),
        "Y_PSP": per_prompt.Y_PSP.mean(),
        "oracle_minus_psp": (per_prompt.oracle - per_prompt.Y_PSP).mean(),
        "oracle_minus_A": (per_prompt.oracle - per_prompt.Y_A).mean(),
        "oracle_minus_B": (per_prompt.oracle - per_prompt.Y_B).mean(),
    })
    table = pd.DataFrame(rows)
    overall = table[table.fold.eq("overall")].iloc[0]
    gate = {
        "mean_oracle": float(overall.oracle),
        "mean_Y_A": float(overall.Y_A),
        "mean_Y_B": float(overall.Y_B),
        "mean_Y_PSP": float(overall.Y_PSP),
        "oracle_headroom_over_psp": float(overall.oracle_minus_psp),
        "oracle_headroom_over_A": float(overall.oracle_minus_A),
        "oracle_headroom_over_B": float(overall.oracle_minus_B),
        "pass": bool(overall.oracle_minus_psp > 0),
    }
    return table, gate


def main() -> None:
    frame = pd.read_parquet(POOLS)
    table, oracle_gate = oracle_report(frame)
    table.to_csv(CAL / "oracle_headroom.csv", index=False)
    (CAL / "oracle_headroom.json").write_text(json.dumps(oracle_gate, indent=2) + "\n")
    print(json.dumps({"oracle_gate": oracle_gate}, indent=2))

    if not oracle_gate["pass"]:
        write_report(frame, None, None, table, oracle_gate, None, None, stopped="no_headroom")
        (CAL / "STOP_NO_HEADROOM").write_text(json.dumps(oracle_gate, indent=2) + "\n")
        print("STOP: the action family {A,B} has no oracle headroom over PSP.")
        return

    features = frame[list(FEATURES)].to_numpy()
    target = frame.V.to_numpy()
    groups = frame.prompt_id.to_numpy()
    folds = sorted(frame.fold.unique().tolist())
    assert folds == list(range(N_FOLDS))

    oof = {}
    model_specs = {
        "ambiguity_g2": lambda: AmbiguityBaseline("g2"),
        "ambiguity_min_g2_g4": lambda: AmbiguityBaseline("min_g2_g4"),
        "ridge": lambda: RidgeModel(),
        "boost": lambda: BoostedModel(),
    }
    for name in model_specs:
        oof[name] = np.full(len(frame), np.nan)
    for fold in folds:
        train = frame.fold.ne(fold).to_numpy()
        test = ~train
        for name, factory in model_specs.items():
            model = factory().fit(features[train], target[train])
            oof[name][test] = model.predict(features[test])

    metrics_rows = []
    fold_policy_rows = []
    prompt_ids = frame.prompt_id.to_numpy()
    y_a = frame.Y_A.to_numpy()
    y_b = frame.Y_B.to_numpy()
    y_psp = frame.Y_PSP.to_numpy()
    fold_values = frame.fold.to_numpy()
    for name in model_specs:
        pred = oof[name]
        metrics_rows.append({"model": name, **regression_metrics(pred, target)})
        picked = choose_b(pred, 0.0)
        y_policy = np.where(picked, y_b, y_a)
        agg = prompt_aggregate(y_policy, prompt_ids)
        a_agg = prompt_aggregate(y_a, prompt_ids)
        b_agg = prompt_aggregate(y_b, prompt_ids)
        psp_agg = prompt_aggregate(y_psp, prompt_ids)
        per_prompt = pd.DataFrame({
            "prompt_id": agg["prompt_ids"],
            "adaptive": agg["per_prompt"],
            "Y_A": a_agg["per_prompt"],
            "Y_B": b_agg["per_prompt"],
            "Y_PSP": psp_agg["per_prompt"],
        }).merge(frame[["prompt_id", "fold"]].drop_duplicates(), on="prompt_id")
        fold_rows = []
        for fold, group in per_prompt.groupby("fold"):
            fold_rows.append({
                "model": name,
                "fold": int(fold),
                "adaptive": group.adaptive.mean(),
                "Y_A": group.Y_A.mean(),
                "Y_B": group.Y_B.mean(),
                "Y_PSP": group.Y_PSP.mean(),
                "delta_vs_psp": (group.adaptive - group.Y_PSP).mean(),
                "delta_vs_A": (group.adaptive - group.Y_A).mean(),
                "delta_vs_B": (group.adaptive - group.Y_B).mean(),
            })
        fold_table = pd.DataFrame(fold_rows)
        fold_policy_rows.append(fold_table)
        record = {
            "model": name,
            "mean_adaptive": float(per_prompt.adaptive.mean()),
            "mean_Y_A": float(per_prompt.Y_A.mean()),
            "mean_Y_B": float(per_prompt.Y_B.mean()),
            "mean_Y_PSP": float(per_prompt.Y_PSP.mean()),
            "mean_delta_vs_psp": float((per_prompt.adaptive - per_prompt.Y_PSP).mean()),
            "mean_delta_vs_A": float((per_prompt.adaptive - per_prompt.Y_A).mean()),
            "mean_delta_vs_B": float((per_prompt.adaptive - per_prompt.Y_B).mean()),
            "positive_psp_folds": int((fold_table.delta_vs_psp >= 0).sum()),
            "beats_A_mean": bool(per_prompt.adaptive.mean() > per_prompt.Y_A.mean()),
            "beats_B_mean": bool(per_prompt.adaptive.mean() > per_prompt.Y_B.mean()),
        }
        metrics_rows[-1].update(record)
        fold_policy_rows[-1] = fold_table

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(CAL / "regret_metrics.csv", index=False)
    pd.concat(fold_policy_rows, ignore_index=True).to_csv(CAL / "fold_results.csv", index=False)
    pd.DataFrame({
        "prompt_id": prompt_ids,
        "fold": fold_values,
        "V": target,
        **{f"V_hat_{name}": oof[name] for name in model_specs},
        "Y_A": y_a,
        "Y_B": y_b,
        "Y_PSP": y_psp,
    }).to_parquet(CAL / "oof_predictions.parquet", index=False)

    primary = metrics[metrics.model.eq(PRIMARY_MODEL)].iloc[0]
    learned_gate = {
        "primary_model": PRIMARY_MODEL,
        "mean_delta_vs_psp": float(primary.mean_delta_vs_psp),
        "condition_adaptive_vs_psp_positive": bool(primary.mean_delta_vs_psp > 0),
        "beats_fixed_A": bool(primary.beats_A_mean),
        "beats_fixed_B": bool(primary.beats_B_mean),
        "condition_adaptive_beats_max_fixed": bool(primary.beats_A_mean and primary.beats_B_mean),
        "positive_psp_folds": int(primary.positive_psp_folds),
        "preferred_at_least_4_of_5": bool(primary.positive_psp_folds >= PREFERRED_MIN_POSITIVE_PSP_FOLDS),
    }
    learned_gate["pass"] = bool(
        learned_gate["condition_adaptive_vs_psp_positive"]
        and learned_gate["condition_adaptive_beats_max_fixed"]
    )
    (CAL / "selectorgate.json").write_text(json.dumps(learned_gate, indent=2) + "\n")

    # Section 29/30 analyses for the primary model.
    v_hat = oof[PRIMARY_MODEL]
    quantile = pd.qcut(v_hat, 5, labels=False, duplicates="drop")
    quintile = pd.DataFrame({"quantile": quantile, "V": target}).groupby("quantile").V.agg(["mean", "count"])
    quintile.to_csv(CAL / "voi_quintiles.csv")
    picked = choose_b(v_hat, 0.0)
    ambiguity = pd.DataFrame({
        "chosen": np.where(picked, "B", "A"),
        "g2": frame.g2.to_numpy(),
        "g4": frame.g4.to_numpy(),
        "top4_std": frame.top4_std.to_numpy(),
        "std": frame["std"].to_numpy(),
        "V": target,
    }).groupby("chosen")[["g2", "g4", "top4_std", "std", "V"]].mean()
    ambiguity.to_csv(CAL / "ambiguity_analysis.csv")

    make_plots(oof, target, metrics, pd.concat(fold_policy_rows, ignore_index=True))

    frozen = None
    if oracle_gate["pass"] and learned_gate["pass"]:
        final = BoostedModel().fit(features, target)
        policy_dir = EXP / "policy"
        policy_dir.mkdir(exist_ok=True)
        import joblib

        joblib.dump(final, policy_dir / "voi_selector.pkl")
        (policy_dir / "feature_config.json").write_text(json.dumps({"features": list(FEATURES)}, indent=2) + "\n")
        frozen = {
            "model": "HistGradientBoostingRegressor(max_leaf_nodes=15, max_iter=200, lr=0.06, l2=1.0)",
            "target": "V = Y_B - Y_A",
            "policy": "choose B iff V_hat > 0",
            "action_A_NFE": 256,
            "action_B_NFE": 256,
            "psp_NFE": 256,
            "oracle_gate": oracle_gate,
            "learned_gate": learned_gate,
        }
        (policy_dir / "policy_config.json").write_text(json.dumps(frozen, indent=2) + "\n")
        (policy_dir / "calibration_checksum.txt").write_text(
            __import__("hashlib").sha256(POOLS.read_bytes()).hexdigest() + "\n"
        )

    write_report(frame, metrics, learned_gate, table, oracle_gate, quintile, ambiguity, stopped=None, frozen=frozen)
    date = __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    (CAL / "COMPLETE").write_text(date + "\n")
    print(json.dumps({"oracle_gate": oracle_gate, "learned_gate": learned_gate, "frozen": frozen}, indent=2))


def make_plots(oof, target, metrics, fold_table) -> None:
    plots = CAL / "plots"
    plots.mkdir(exist_ok=True)
    v_hat = oof[PRIMARY_MODEL]
    sample = np.random.default_rng(0).integers(0, len(target), size=min(20000, len(target)))
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(v_hat[sample], target[sample], s=2, alpha=0.15)
    limit = float(np.quantile(np.abs(target), 0.995))
    plt.plot([-limit, limit], [-limit, limit], "r--", linewidth=1)
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.xlabel("predicted V")
    plt.ylabel("realized V")
    plt.title("VOI calibration (OOF)")
    plt.tight_layout()
    plt.savefig(plots / "voi_predicted_vs_realized.png", dpi=120)
    plt.close()
    quint = pd.DataFrame({"q": pd.qcut(v_hat, 5, labels=False, duplicates="drop"), "V": target}).groupby("q").V.mean()
    plt.figure(figsize=(5, 3))
    plt.bar(quint.index.astype(str), quint.values)
    plt.axhline(0, color="k", linewidth=0.8)
    plt.xlabel("predicted VOI quintile")
    plt.ylabel("realized V")
    plt.title("Realized VOI by predicted VOI")
    plt.tight_layout()
    plt.savefig(plots / "voi_quintiles.png", dpi=120)
    plt.close()
    primary_folds = fold_table[fold_table.model.eq(PRIMARY_MODEL)]
    for name, label in (("delta_vs_psp", "Adaptive - PSP"), ("delta_vs_A", "Adaptive - A"), ("delta_vs_B", "Adaptive - B")):
        plt.figure(figsize=(5, 3))
        plt.bar(primary_folds.fold.astype(str), primary_folds[name])
        plt.axhline(0, color="k", linewidth=0.8)
        plt.xlabel("fold")
        plt.ylabel("mean reward delta")
        plt.title(label)
        plt.tight_layout()
        plt.savefig(plots / f"fold_{name}.png", dpi=120)
        plt.close()


def write_report(frame, metrics, learned_gate, oracle_table, oracle_gate, quintile, ambiguity, stopped, frozen=None) -> None:
    oracle_rows = "\n".join(
        f"| {row.fold} | {row.oracle:.6f} | {row.Y_A:.6f} | {row.Y_B:.6f} | {row.Y_PSP:.6f} | "
        f"{row.oracle_minus_A:+.6f} | {row.oracle_minus_B:+.6f} | {row.oracle_minus_psp:+.6f} |"
        for row in oracle_table.itertuples()
    )
    if stopped == "no_headroom":
        outcome = (
            "**STOP: the action family {A,B} does not contain enough headroom to beat PSP.** "
            "Mean OOF Oracle - PSP = "
            f"{oracle_gate['oracle_headroom_over_psp']:+.6f} <= 0, so no learned selector (which cannot "
            "exceed its oracle) and no GenEval validation were run. The bottleneck is action-family "
            "expressiveness, not predictor quality."
        )
        body = ""
    else:
        outcome = (
            "Oracle headroom exists. Learned-policy gate: "
            f"**{'PASS' if learned_gate['pass'] else 'FAIL'}** "
            f"(Adaptive - PSP = {learned_gate['mean_delta_vs_psp']:+.6f}, beats A: "
            f"{learned_gate['beats_fixed_A']}, beats B: {learned_gate['beats_fixed_B']}, positive PSP folds "
            f"{learned_gate['positive_psp_folds']}/5)."
        )
        model_rows = "\n".join(
            f"| {row.model} | {row.mae:.4f} | {row.rmse:.4f} | {row.spearman:.3f} | {row.sign_accuracy:.3f} | "
            f"{row.precision_V_pos:.3f} | {row.recall_V_pos:.3f} | {row.mean_delta_vs_psp:+.6f} | "
            f"{row.mean_delta_vs_A:+.6f} | {row.mean_delta_vs_B:+.6f} |"
            for row in metrics.itertuples()
        )
        quintile_rows = "\n".join(f"| {int(row.Index) + 1} | {row.mean:+.6f} | {int(row.count)} |" for row in quintile.itertuples())
        ambiguity_rows = "\n".join(
            f"| {row.Index} | {row.g2:.4f} | {row.g4:.4f} | {row.top4_std:.4f} | {row.std:.4f} | {row.V:+.6f} |"
            for row in ambiguity.itertuples()
        )
        body = f"""
## Selector metrics and OOF policy

| Model | MAE | RMSE | Spearman | sign acc | precision V>0 | recall V>0 | Delta vs PSP | Delta vs A | Delta vs B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{model_rows}

## Realized VOI by predicted VOI quintile (primary model)

| Quintile | mean realized V | pools |
|---:|---:|---:|
{quintile_rows}

## Ambiguity analysis (primary model)

| Chosen | mean g2 | mean g4 | mean top4_std | mean std | mean V |
|---|---:|---:|---:|---:|---:|
{ambiguity_rows}
"""
    report = f"""# Value-of-Information calibration report

## Outcome

{outcome}

Two equal-compute actions at step 16: A = 10->2@16->final (256 NFE) and
B = 10->4@16->1@32->final (256 NFE). PSP = 8->4@16->2@32 (256 NFE) uses the first
8 of the ordered pool. Label `V = Y_B - Y_A`; features are step-16 score geometry
only. Pools are prompt-grouped and metrics aggregate within prompt before
averaging, so hundreds of pools per prompt do not fake sample size.

## Oracle headroom (computed before any training)

| Fold | Oracle | A | B | PSP | Oracle-A | Oracle-B | Oracle-PSP |
|---|---:|---:|---:|---:|---:|---:|---:|
{oracle_rows}

Headroom over PSP is the mandatory first gate. Oracle-A measures A/B
heterogeneity; Oracle-B measures how much A adds on top of B.
{body}

## Interpretation guardrails

This is calibration evidence only. HPS and official GenEval were never inspected
and never influenced the selector. If the gate failed, no parameter was tuned in
response and a broader action family requires a new development cycle.
"""
    (CAL / "CALIBRATION_REPORT.md").write_text(report)


if __name__ == "__main__":
    main()
