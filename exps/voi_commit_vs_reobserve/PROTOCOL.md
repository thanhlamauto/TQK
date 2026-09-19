# Value-of-Information: commit vs re-observe (protocol, frozen before analysis)

Two equal-compute actions available after scoring 10 candidates at step 16:

* **Action A (commit now):** `10 -> 2 @16 -> final` = `160 + 2*48 = 256` NFE.
  Keep Top-2 by step-16 ImageReward; `Y_A = max final` over them.
* **Action B (buy information):** `10 -> 4 @16 -> 1 @32 -> final`
  = `160 + 4*16 + 1*32 = 256` NFE. Keep Top-4 by step-16, then keep the single
  highest step-32 ImageReward; `Y_B` is its final reward.
* **PSP (baseline):** `8 -> 4 @16 -> 2 @32 -> final` = 256 NFE, applied to the
  **first 8** candidates of the ordered pool so initial pools are nested.

The label is `V = Y_B - Y_A`; the oracle is `max(Y_A, Y_B)`.  These definitions
are not changed and no third action is added in this experiment.

## Calibration data

* Independent non-GenEval corpus: the 200 ImageReward prompts of
  `exps/dense_calibration_200` (selection seed 20260920, zero exact GenEval
  overlap).  Trajectories are reused, not regenerated; only `IR_step16`,
  `IR_step32`, `IR_final` are used.
* Ordered 10-candidate pools sampled without replacement, 1000 per prompt, RNG
  seed 20260920 (per-prompt generator `[20260920, prompt_id]`, row-argsort of a
  uniform matrix).  PSP uses the first 8 of each pool.

## Features (step-16 only)

For `s1 >= ... >= s10`, `z_i = (s_i - median(s)) / (MAD(s) + 1e-6)`:
`z1..z10`; `g1..g4 = z_i - z_{i+1}`; `z1-z3`, `z1-z4`, `z1-z5`; `mean`, `std`,
`MAD`, `max-min`, `top2_mean`, `top4_mean`, `top4_std`.  No step-32 scores, final
rewards, prompt category, HPS or GenEval are inputs.

## Predictors (predeclared, low capacity)

* Ambiguity baseline: isotonic regression of `V` on the scalar `g2`
  (`min(g2,g4)` also reported).
* Ridge on standardized features (`alpha=1.0`).
* `HistGradientBoostingRegressor(max_leaf_nodes=15, max_iter=200, lr=0.06,
  l2=1.0, min_samples_leaf=40)` - **primary** model, predeclared.
* Policy: choose B iff `V_hat > 0` (single scalar threshold fixed at 0; primary
  policy stays simple).  `V` is predicted directly (delta prediction).

## Grouped cross validation

5 prompt groups of 40 (frozen fold labels).  All pools from one prompt stay in
one fold; predictions are out-of-fold.  Regression metrics (MAE, RMSE, Spearman),
decision metrics (sign accuracy, precision/recall for `V>0`), and the actual
policy reward `Y_policy = Y_B if choose B else Y_A`.

## Prompt-level aggregation

The independent unit is the calibration prompt: every metric aggregates within
prompt first, then averages over prompts.  Row-level p-values treating pools as
independent are never reported.

## Gates

**Gate 1 (mandatory, before training):** mean Out-of-Fold `Oracle - PSP > 0`.
The oracle is computed first; if it fails, STOP, train nothing, launch no GenEval,
and report that the bottleneck is the action family.

**Gate 2 (only if Gate 1 passes):**
* mean OOF `Adaptive - PSP > 0`, and
* mean OOF `Adaptive > max(mean A, mean B)`.

Preferred: `Adaptive - PSP >= 0` in at least 4 of 5 folds.

If Gate 2 fails, stop; report whether the bottleneck is predictor quality
(Gate 1 passed but Gate 2 failed) or action-family expressiveness (Gate 1 failed).

## Frozen validation (only if both gates pass)

Fit the primary model on all 200 prompts, freeze features/model/threshold, then
compare PSP, Fixed A, Fixed B and Adaptive on all 553 GenEval prompts with a
fresh pre-recorded seed base, online direct inference, same GPU per prompt, 256
NFE per action, primary metric ImageReward, secondary HPS and GenEval, 10,000
prompt-level paired bootstrap resamples.
