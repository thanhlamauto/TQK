# Adaptive-K calibration report

## Outcome

**OOF gate FAILED.** Per the pre-registered stop rule, no predictor was frozen and no GenEval validation was launched. Failure is reported, not tuned away.

Method studied: 10 initial seeds -> step 16 -> observe the 10 step-16 ImageReward
scores -> choose K in {1,2,3} -> finish K candidates -> final winner by final
ImageReward.  Cost is `160 + 48K` per prompt, and the exact-budget policy enforces
`sum_p K_p = 2N`, giving exactly 256 average logical UNet evaluations per prompt.

## Required calibration table

| Fold | PSP IR | Fixed 10->2 IR | Adaptive-K IR | Delta Adaptive-PSP | Delta Adaptive-Fixed | K=1/2/3 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0.944637 | 0.864306 | 0.876353 | -0.068285 | +0.012047 | 0.178/0.643/0.178 |
| 1 | 0.869731 | 0.820168 | 0.821401 | -0.048330 | +0.001233 | 0.224/0.551/0.224 |
| 2 | 0.822599 | 0.788914 | 0.796629 | -0.025970 | +0.007715 | 0.259/0.482/0.259 |
| 3 | 0.969912 | 0.936401 | 0.932492 | -0.037420 | -0.003909 | 0.288/0.423/0.288 |
| 4 | 0.663664 | 0.600761 | 0.604086 | -0.059578 | +0.003326 | 0.174/0.651/0.174 |

Mean Delta Adaptive-PSP = -0.047917; mean Delta
Adaptive-fixed10->2 = +0.004082.

## Regret-predictor metrics (held-out prompts)

| Model | MAE K=1 | MAE K=2 | MAE K=3 | RMSE K=1 | RMSE K=2 | RMSE K=3 | Spearman K=1 | Spearman K=2 | Spearman K=3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Model A isotonic (projected) | 0.3544 | 0.2467 | 0.1834 | 0.4581 | 0.3363 | 0.2731 | 0.048 | 0.046 | 0.005 |
| Model B boosted (unprojected) | 0.3195 | 0.2240 | 0.1687 | 0.4331 | 0.3196 | 0.2645 | 0.344 | 0.296 | 0.257 |
| Model B boosted (projected) | 0.3196 | 0.2240 | 0.1685 | 0.4332 | 0.3195 | 0.2644 | 0.343 | 0.295 | 0.257 |

Monotone projection: Model B MAE K=1/2/3 changes from
0.3195/0.2240/0.1687
to
0.3196/0.2240/0.1685.

## Marginal-benefit metrics (held-out)

| Model | MAE b12 | MAE b23 | Spearman b12 | Spearman b23 |
|---|---:|---:|---:|---:|
| Model B projected | 0.2180 | 0.1155 | 0.129 | 0.111 |
| Model A isotonic | 0.2223 | 0.1184 | 0.069 | 0.029 |

## OOF gate

| Condition | Value | Pass |
|---|---:|---|
| A: mean Delta Adaptive-PSP > 0 | -0.047917 | False |
| B: mean Delta Adaptive-fixed10->2 > 0 | +0.004082 | True |
| C: >=4/5 folds Delta Adaptive-fixed10->2 >= 0 | 4/5 | True |
| Preferred: >=3/5 folds positive vs PSP | 0/5 | False |
| **Overall gate** |  | **False** |

Secondary (Model A isotonic) mean Delta vs PSP = -0.051694,
vs fixed10->2 = +0.000305.

## Chosen-K accounting

Each fold's exact-budget allocation satisfies `n1 + 2 n2 + 3 n3 = 2N` and hence
`n1 = n3` (verified per batch draw in `policy_simulation`).  Mean K per prompt is
2.0 by construction.

## Feature importance (Model B, averaged over folds)

| Feature | Importance |
|---|---:|
| maxmin | 0.4646 |
| top3_mean | 0.3136 |
| top3_std | 0.0625 |
| std | 0.0441 |
| mad | 0.0393 |
| g1 | 0.0273 |
| z1z3 | 0.0132 |
| z2 | 0.0092 |
| g2 | 0.0081 |
| z1 | 0.0051 |

## Interpretation guardrails

This is calibration evidence only.  HPS and official GenEval were not inspected.
If the gate failed, no online GenEval validation was run and no parameter was
tuned in response.  A new policy requires a new development cycle.
