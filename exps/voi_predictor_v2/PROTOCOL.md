# Predictor v2 protocol (frozen before analysis)

Same two equal-compute actions as the VOI experiment:

* A: `10 -> 2 @16 -> final` (256 NFE)
* B: `10 -> 4 @16 -> 1 @32 -> final` (256 NFE)
* PSP: `8 -> 4 @16 -> 2 @32 -> final` (256 NFE) on the first 8 of the ordered pool.

## Objective change

Instead of MSE-regressing `V = Y_B - Y_A`, train a **weighted classifier** for
`z = 1[V > 0]` with weight `w = min(|V|, q95(|V|))`, computed on training folds.
A Bayes-optimal weighted classifier chooses B iff `E[V | X] > 0`, aligning the
training loss with final ImageReward.  The decision threshold `delta*` is chosen
on training folds only (`argmax_delta Q_policy`), then frozen on the held-out
fold.

## Feature families (exactly four; no fifth added after seeing results)

* **F0** - scalar step-16 geometry: `z1..z10`, `g1..g4`, `z1-z3`, `z1-z4`,
  `z1-z5`, `mean`, `std`, `MAD`, `max-min`, `top2_mean`, `top4_mean`, `top4_std`.
* **F1** - F0 plus temporal reward dynamics from steps 12/14/16:
  `g_k^{12}, g_k^{14}`, boundary-gap velocities `g_k^{16}-g_k^{14}` and
  `g_k^{14}-g_k^{12}`, per-rank velocities `v_i = s_i^{16}-s_i^{14}` and
  accelerations `a_i` for ranks 1-4, `m_rescue = max v(rank3,4) - min v(rank1,2)`,
  Kendall `tau(12,16)` and `tau(14,16)`, top-4 churn 12->16 and 14->16,
  `mean/std` at 12 and 14.  No extra UNet NFE; only extra ImageReward calls.
* **F2** - F0 plus predicted-clean latent drift `||x0_16 - x0_14|| / ||x0_16||`
  summaries (mean/max top-4, rank3/4 minus rank1/2, pairwise top-4 diversity).
  No extra verifier calls.  Requires a dynamics bank.
* **F3** - F1 + latent drift + frozen ImageReward hidden features at step 16
  (PCA to 16 dimensions, fit on training folds only).

F0/F1 are computed from the existing calibration bank.  F2/F3 require a new
dynamics bank that stores `x0_hat` at steps 14 and 16 (and, for F3,
ImageReward hidden states).

## Grouped cross validation and metrics

Prompt-grouped 5 folds.  For each family the primary model is the one with the
highest OOF policy value `Q_policy = E[Y_B 1(pi=B) + Y_A 1(pi=A)]` (threshold
frozen per fold).  Report `Regret = Q_oracle - Q_policy` and the pre-registered
capture

```
H = (Q_policy - max(Q_A, Q_B)) / (Q_oracle - max(Q_A, Q_B)).
```

PSP requires `H >= 62%`.  Also report captured positive reward mass, incurred
negative mass, and precision/recall on the largest positive-V pools.

## Gate

Launch full GenEval validation only if, out of fold:

1. mean `Adaptive - PSP > 0`;
2. at least 4 of 5 folds non-negative;
3. `H > 65%`.

`H >= 70%` is the strong bar.  If no family passes, stop; the conclusion
distinguishes insufficient dynamics information from predictor failure, and the
next direction is a noise-aware verifier (e.g. TTSnap) rather than a more complex
predictor.
