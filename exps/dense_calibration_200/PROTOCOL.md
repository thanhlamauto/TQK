# Dense calibration protocol (frozen before any calibration inference)

This file operationalises the pre-registered next experiment in the repository
`README.md` (sections 1-6).  Every constant below was fixed before the new
calibration rewards were inspected.

## Fixed generation protocol

- `runwayml/stable-diffusion-v1-5`, deterministic DDIM, 64 steps, `eta=0`,
  fp16, guidance scale 7.5.
- ImageReward is the only calibration/pruning reward; HPS and official GenEval
  are never computed during calibration.
- Budget: `M*t + K*(64-t) <= 256`; `M(t,K)=floor((256-K*(64-t))/t)`.
- Search exactly `t in {10,...,28}`, `K in {1,2,3}` = 57 policies.
- The trajectory bank stores ImageReward at every step in `{10..28}` (the search
  grid) and, additionally, step 32.  Step 32 is included only so the fixed
  multi-stage PSP comparator `8->4@16->2@32->1@64` can be replayed on the same
  bank and the same subset draws; it is not in the single-stage search space and
  cannot be selected.

## Independent calibration bank

- Prompt source: `Fk-Diffusion-Steering/text_to_image/prompt_files/test_ir.json`.
- Exactly 200 unique prompts, selection seed `20260920`; zero exact (casefolded)
  overlap with the 553 official GenEval prompts is asserted.
- 25 independent complete 64-step trajectories per prompt, candidate seed
  `20260920 + prompt_id*32 + candidate_id`.
- Stored quantities: `X[p,i,t]` for `t in {10..28,32}` and `Y[p,i]` at step 64.
- Five-fold split seed `20260920`, 40 prompts per fold.

## Breadth value and pruning regret

- Exact `O_p(M) = sum_{j=M..25} Y_(j) * C(j-1,M-1) / C(25,M)`, `O(M)=mean_p O_p(M)`.
- For each required pool size (the distinct policy `M` values plus PSP's 8), 500
  subsets per prompt drawn with subset seed `20260920`.  For pool size `M` and
  prompt index `p` the generator is seeded by `[20260920, M, p]` and subsets are
  the first `M` columns of the row-ranks of a uniform `(500,25)` matrix.  The
  same subsets are reused for every `(t,K)` with that `M`.
- `p_miss`, `ell = E[regret | miss]`, `R = p_miss*ell`, `Q = O(M) - R`, and
  `Delta = Q(single-stage) - Q(PSP)`.  The invariant `mean regret = p_miss*ell`
  is asserted to `< 1e-9`.
- PSP is replayed on the size-8 subset draws.

## Shared reliability surfaces

- `p_miss` logistic model:

      logit(p_miss) = b0 + b1*log(M) + b2*(K/M) + sum_j w_j * max(0, knot_j - q)

  with `q = t/64`, knots `(0.20, 0.25, 0.30, 0.35, 0.40)`, `w_j >= 0` (so
  `p_miss` is non-increasing in denoising depth), L2 = `1e-3` on `w_j`.
- `ell` Gamma regression with log link:

      ell = exp(c0 + c1*log(M) + c2*(K/M) + c3*q + c4*q^2)

  L2 = `1e-3` on `c1..c4`; target floored at `1e-4`.
- Both fit by weighted deviance minimisation (binomial / Gamma), L-BFGS-B,
  coefficient bounds `[-30,30]` (plus `w_j >= 0`), weights `n` (`p_miss`) and
  `n_miss` (`ell`).

## Five-fold out-of-fold gate

- One deterministic five-fold split of the 200 prompts (fold seed `20260920`).
- For each fold: fit surfaces on the 160 training prompts; select the policy with
  the largest predicted Delta on the 160; replay it and PSP on the 40 held-out
  prompts; record the held-out paired Delta.
- Predicted vs actual ranking of all 57 policies on the held-out fold is
  summarised by Spearman correlation; the fraction of folds whose predicted
  winner is in the actual top-3/top-5 is reported.
- Pre-declared gate (all must hold):
  1. mean held-out Delta ImageReward `> 0`;
  2. at least 4 of 5 folds positive;
  3. mean schedule-rank Spearman `>= 0.80`;
  4. coherent neighbourhood: selected checkpoints span `<= 4` and selected `K`
     span `<= 1`.

## Freeze gate

- Only if the OOF gate passes: refit on all 200 prompts, bootstrap prompts with
  1,000 fixed-seed (`20260920`) resamples, and compute the 80% lower confidence
  bound (10th percentile) of predicted Delta ImageReward for every policy.
- Freeze `argmax LCB80`.  Proceed to the confirmatory 553 run only if the winning
  `LCB80 > 0`; otherwise stop and report failure without running GenEval-553.
- The threshold is not lowered after seeing the result and no substitute schedule
  is used.

## Stop rule

If the OOF gate or the freeze gate fails, `run_all.sh` stops before the
confirmatory 553 phase and writes the failure in `CALIBRATION_REPORT.md`.
