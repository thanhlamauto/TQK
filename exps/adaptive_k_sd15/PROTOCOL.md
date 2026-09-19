# Adaptive-K Screen-and-Commit: protocol (frozen before calibration analysis)

Method: 10 initial seeds -> denoise all 10 to step 16 -> observe the 10 step-16
ImageReward scores -> choose `K in {1,2,3}` -> finish those K candidates ->
select the final winner by final ImageReward.  This development cycle fixes
`M=10`, `t=16`, `K in {1,2,3}` and does not tune them.

## Compute rule

Per prompt: `cost(K) = 10*16 + K*(64-16) = 160 + 48K` -> K=1:208, K=2:256,
K=3:304.  For an evaluation batch of N prompts the exact-budget policy enforces
`sum_p K_p = 2N`, so total logical UNet evaluations are exactly `256N` and the
average is exactly 256/prompt.  This identity implies `n1 = n3`, which is
asserted for every simulated batch.

## Calibration data

* Independent non-GenEval corpus: the 200 ImageReward prompts of
  `exps/dense_calibration_200` (selection seed 20260920, zero exact GenEval
  overlap).  GenEval-553 is never used for calibration.
* 25 seeds/prompt; the source trajectories are deterministic DDIM, T=64, eta=0,
  fp16, guidance 7.5, candidate seed `20260920 + prompt_id*32 + candidate_id`.
* Only ImageReward at step 16, step 32 and step 64 is extracted into
  `calibration/bank.parquet`.  No HPS/GenEval and no other checkpoints.

## 10-seed pools and labels

* For each prompt, draw 1000 subsets of size 10 with RNG seed 20260919
  (per-prompt generator seeded by `[20260919, prompt_id]`).
* Sort each subset's step-16 scores descending (candidate-id tie-break) and
  define `L_K = max_i final_i - max_{i in TopK(step16)} final_i` for K=1,2,3.
  `L1 >= L2 >= L3 >= 0` holds by construction; the empirical violation rate is
  reported.
* Baseline replay on each subset: fixed 10->1 / 10->2 / 10->3, and fixed PSP
  `8->4@16->2@32`.  PSP uses the 8 lowest-candidate-id members of the subset
  (mirroring the online "candidates 0..7"), keeping 4 by step 16, 2 by step 32,
  then the final winner.

## Features (step-16 only)

For sorted scores `s1>=...>=s10`, with `median_s`, `MAD_s = median(|s-median|)+1e-6`
and `z_i=(s_i-median_s)/MAD_s`: `g1,g2,g3 = z1-z2, z2-z3, z3-z4`; `z1..z4`;
`z1-z3`; `z1-z4`; `std(s)`; `MAD(s)`; `max-min`; `top3_mean`; `top3_std`.  No
final reward, prompt label, GenEval outcome, HPS or future-timestep information
is used.

## Regret predictors

* Model A (baseline): one 1-D isotonic (decreasing) regression per K on the
  boundary gap `g_K`.
* Model B (primary, chosen before inspecting labels): one
  `GradientBoostingRegressor` per K on all features
  (`n_estimators=200, max_depth=3, lr=0.05, min_samples_leaf=50`).
* Both clip predictions to `>= 0`.  A monotone projection
  `Lhat1 >= Lhat2 >= Lhat3 >= 0` (exact PAVA on clipped values) is applied; the
  effect on held-out MAE is reported.

## Prompt-grouped cross validation

* 5 folds of 40 prompts (folds frozen in the prompt manifest); every subset from
  a prompt stays in the same fold.
* Metrics on each held-out fold for K=1,2,3: MAE, RMSE,
  Spearman(predicted, realized); plus marginal-benefit MAE/Spearman for
  `b12 = L1-L2` and `b23 = L2-L3`.

## Exact-budget policy simulation and OOF evaluation

* Each fold's held-out prompts have 1000 simulated pools.  The OOF policy is
  evaluated over 1000 paired batch draws: draw `b` uses subset `b` for every
  held-out prompt, giving one 10-pool per prompt per batch.
* Per draw, `K_p` is chosen by exact dynamic programming minimising
  `sum_p Lhat_{K_p}(X_p)` subject to `K_p in {1,2,3}` and `sum_p K_p = 2N`
  (N=40).  The DP is verified against brute force on small instances.  The
  resulting `n1=n3` check is asserted.
* Realized rewards use only step-16 information and the actual finals: adaptive
  reward is `max final over TopK_p`, fixed 10->1/2/3 use Top1/2/3, PSP uses the
  subset's first 8 seeds.  Deltas are paired within each draw.

## OOF gate

Hard conditions (all required):

* A: mean OOF `Delta IR` Adaptive - PSP `> 0`;
* B: mean OOF `Delta IR` Adaptive - fixed 10->2 `> 0`;
* C: at least 4 of 5 held-out folds have `Delta IR` Adaptive - fixed 10->2 `>= 0`.

Preferred (reported, not required): at least 3 of 5 folds positive vs PSP.
If the gate fails, STOP, write the calibration report, and do not launch GenEval.
No feature, K set, M, checkpoint or gate may be changed after seeing GenEval.

## Frozen validation (only if the gate passes)

Train the regret predictor on all 200 prompts and save under `policy/`.  Compare
PSP, fixed 10->2 and exact-budget Adaptive-K on all 553 GenEval prompts with
fresh pre-recorded seed bases, online (not replay), three repetitions, same GPU
per prompt, primary metric ImageReward, secondary HPS and official GenEval,
10,000 prompt-level paired bootstrap resamples.
