# TQK: calibrated single-stage pruning for diffusion inference

This repository contains the reproducible SD1.5 experiments used to compare a
single screen-and-commit pruning event with Progressive Seed Pruning (PSP) at a
fixed inference-time compute budget.

The implementation is derived from the official
[`rogerioagjr/PSP`](https://github.com/rogerioagjr/PSP) repository for
*Inference-Time Scaling of Diffusion Models via Progressive Seed Pruning*
(ECCV 2026). The original authors are Rogério Guimarães and Pietro Perona. The
upstream MIT license is retained in [`LICENSE`](LICENSE).

## What is included

```text
TQK/
├── Fk-Diffusion-Steering/text_to_image/   # PSP pipeline and reward helpers
├── geneval/                               # official GenEval evaluator
├── setup/                                 # SD1.5 and GenEval setup scripts
└── exps/
    ├── direct_sd15_psp_vs_10to2_300/      # development/reference run
    ├── direct_sd15_psp_vs_10to2_553/      # all-553 reference run
    ├── single_stage_calibration/           # independent calibration bank
    └── single_stage_validation_553/        # frozen confirmatory comparison
```

Large trajectory banks, generated images, model caches, credentials, and
compressed result bundles are intentionally excluded. The frozen schedule,
protocol manifest, prompt selections, source code, and compact Markdown reports
needed to audit the completed experiment are committed.

Two correctness constraints are preserved throughout the code:

1. pruning scores the DDIM scheduler's existing `pred_original_sample`;
2. per-particle random generators are pruned together with survivor latents.

## Completed result

The first independent calibration used 120 ImageReward prompts with zero exact
GenEval overlap and selected `9→2@17`. The schedule was frozen before evaluation
on all 553 official GenEval prompts with a fresh candidate-pool seed.

| Method | Logical UNet evals | ImageReward | HPS | GenEval | sec/prompt | peak VRAM | verifier calls/scores |
|---|---:|---:|---:|---:|---:|---:|---:|
| PSP `8→4@16→2@32` | 256 | 0.843532 | 0.277765 | 0.533454 | 5.009 | 8.27 GiB | 3 / 14 |
| Single-stage `9→2@17` | 247 | 0.830516 | 0.278149 | 0.547920 | 4.816 | 8.77 GiB | 2 / 11 |

Paired deltas (single-stage minus PSP):

| Metric | Delta | 95% paired CI |
|---|---:|---:|
| ImageReward | -0.013016 | [-0.038019, 0.012314] |
| HPS | +0.000384 | [-0.000908, 0.001658] |
| GenEval | +0.014467 | [-0.007233, 0.036166] |

All confidence intervals contain zero. The supported claim is therefore an
efficiency trade-off—3.5% fewer UNet evaluations, 3.8% lower latency, and fewer
verifier operations—not a statistically established quality improvement.

See:

- [`exps/single_stage_calibration/PHASE1_REPORT.md`](exps/single_stage_calibration/PHASE1_REPORT.md)
- [`exps/single_stage_validation_553/FINAL_REPORT.md`](exps/single_stage_validation_553/FINAL_REPORT.md)
- [`exps/single_stage_validation_553/FROZEN_SCHEDULE.json`](exps/single_stage_validation_553/FROZEN_SCHEDULE.json)

## Environment

The completed run used Linux, two RTX 4090 GPUs, Python 3.10, Stable Diffusion
1.5, fp16, deterministic DDIM, 64 steps, `eta=0`, and guidance scale 7.5.

```bash
conda create -n psp python=3.10 -y
conda activate psp
bash setup/setup.sh
```

Install the official GenEval environment separately:

```bash
bash setup/setup_geneval.sh
```

Model and package caches are not committed. Configure Hugging Face credentials
through `huggingface-cli login` or an environment secret; never write tokens
into source files.

## Reproducing the completed experiment

The top-level runner performs preflight checks, builds the independent
calibration bank, freezes exactly one schedule, validates it against fixed PSP,
scores final winners with HPS and official GenEval, and writes reports.

```bash
python exps/single_stage_calibration/budget_check.py
python exps/single_stage_validation_553/budget_check.py

# Expensive: 120 prompts × 25 full trajectories, followed by GenEval-553.
bash exps/single_stage_calibration/run_all.sh
```

The runner defaults to paths used on the original machine. Override these when
needed:

```bash
export GEN_VENV=/path/to/psp_sd15_env
export GENEVAL_VENV=/path/to/psp_geneval_env
export GENEVAL_MMDET=/path/to/mmdetection-v2.28.2
export GENEVAL_WEIGHTS=/path/to/geneval_weights
```

The 553-prompt phase may also be inspected independently in
[`exps/single_stage_validation_553/README.md`](exps/single_stage_validation_553/README.md).
Do not modify the committed frozen schedule when reproducing the reported run.

## Pre-registered next experiment

The next experiment tests whether a structured calibration procedure can find
a single-stage allocation expected to beat progressive PSP. It is implemented in
[`exps/dense_calibration_200/`](exps/dense_calibration_200/README.md) (calibration
and freeze gate) and
[`exps/confirmatory_553/`](exps/confirmatory_553/README.md) (confirmatory
validation), and was executed once; see [the result](#result-of-the-next-experiment).
Its search space, gates, endpoints, and thresholds must not be changed after the
new calibration rewards are inspected.

### 1. Fixed generation protocol

- Stable Diffusion 1.5, deterministic DDIM, 64 denoising steps, `eta=0`, fp16,
  and the same guidance and sampler settings as the completed experiment.
- ImageReward is the only calibration/pruning reward. HPS and official GenEval
  are final-validation metrics only.
- Fixed budget: `M*t + K*(64-t) <= 256`.
- `M` is derived, never tuned independently:

  ```text
  M(t,K) = floor((256 - K*(64-t)) / t)
  ```

- Search exactly `t ∈ {10,11,...,28}` and `K ∈ {1,2,3}`: 57 policies.

### 2. Independent calibration bank

Select exactly 200 prompts once from the repository ImageReward corpus using a
new fixed selection seed. Record the prompt IDs and verify zero exact overlap
with all 553 GenEval prompts. For each prompt, run 25 independent complete
trajectories and store:

```text
X[p,i,t] = ImageReward(pred_original_sample at t), t=10..28
Y[p,i]   = ImageReward(final sample at step 64)
```

Do not compute or inspect GenEval-553 ImageReward, HPS, or GenEval while
selecting a schedule.

### 3. Breadth value and pruning regret

For each prompt, sort its 25 final rewards ascending as `Y_(j)`. Compute the
expected oracle value of a uniformly sampled `M`-seed pool exactly:

```text
O_p(M) = sum_{j=M..25} Y_(j) * C(j-1, M-1) / C(25, M)
O(M)   = mean_p O_p(M)
```

For each required pool size, draw 500 subsets per prompt with one fixed subset
seed. Reuse the same subsets for all `(t,K)` comparisons. For every subset:

```text
p_miss = P(final-best seed is not in TopK(X_t))
ell    = E[oracle final reward - best-survivor final reward | miss]
R      = p_miss * ell
Q      = O(M) - R
Delta  = Q(single-stage) - Q(PSP 8→4@16→2@32)
```

Replay PSP on the same bank and subset draws. As an invariant, direct mean
regret must equal `p_miss * ell` to numerical precision.

### 4. Shared reliability model

Do not estimate 57 unrelated cells and choose the largest noisy mean. Fit
low-complexity shared surfaces:

- empirical or monotone-concave `O(M)`;
- logistic/GAM `p_miss(q, log(M), K/M)` with `q=t/64` and miss probability
  constrained not to increase with denoising depth;
- positive regression/GAM `ell(q, log(M), K/M)`.

Document the exact model formula, regularization, software versions, and every
constraint before fitting.

### 5. Five-fold out-of-fold selection

Create one deterministic five-fold split of the 200 prompts. For each fold:

1. fit all surfaces on 160 prompts;
2. select exactly one policy using those 160 prompts only;
3. replay the selected policy and fixed PSP on the 40 held-out prompts;
4. record the paired held-out ImageReward difference;
5. evaluate predicted versus actual ranking of all 57 policies on the held-out
   fold without using that ranking to revise the selector.

Report schedule-rank Spearman correlation and whether the predicted winner is
in the actual top 3/top 5. The out-of-fold gate requires positive mean held-out
`Delta IR`; the pre-declared stability targets are at least four positive folds,
schedule choices in one coherent neighborhood, and rank correlation around
`0.8` or higher. Failure is reported, not tuned away.

### 6. Freeze gate

Only after the out-of-fold gate passes, refit on all 200 prompts. Bootstrap
prompts with 1,000 fixed-seed resamples and compute the 80% lower confidence
bound for every schedule's `Delta IR`. Freeze:

```text
argmax_schedule LCB_80(Delta IR)
```

Proceed only if the winning `LCB_80(Delta IR) > 0`. If no policy passes, stop
without running GenEval-553. Do not lower the 80% threshold after seeing the
result and do not substitute another schedule.

### 7. Confirmatory validation

If and only if the freeze gate passes, compare the frozen schedule with
unchanged PSP on all 553 official GenEval prompts using three fresh,
pre-recorded candidate-pool base seeds disjoint from calibration and every
previous validation run.

For prompt `p` and repetition `r`, compute the paired difference `d[p,r]`.
Average the three repetitions within prompt first, then bootstrap the 553
prompt-level averages 10,000 times. Do not treat the 1,659 images as independent
samples.

Primary endpoint: ImageReward. A strong success requires positive mean
`Delta IR` and a 95% paired-confidence lower bound above zero. HPS and GenEval
are secondary endpoints and must not show clear degradation. If ImageReward
wins while independent metrics decline, restrict the claim to better
optimization of ImageReward.

Always report logical UNet evaluations, runtime mean/median/p90, throughput per
GPU, peak allocated VRAM, batched verifier calls, candidate scores per prompt,
all prompt IDs and seeds, and all paired confidence intervals.

## Result of the next experiment

The denser calibration was run once, exactly as pre-registered above (200 prompts,
57 policies, shared reliability surfaces, deterministic five-fold out-of-fold
selection). The **out-of-fold gate failed**, so per the pre-registered stop rule
the confirmatory 553-prompt phase was **not** run.

- Calibration bank: 200 prompts × 25 complete trajectories. ImageReward stored at
  every step `10..28` plus step `32` (for PSP replay) and step 64.
- Fixed PSP `8→4@16→2@32` replayed on the same subset draws: mean calibration IR
  `0.851613`.
- Across all 57 single-stage policies, the empirical `Delta IR` on the calibration
  bank was **negative for every policy**; the best was `7→2@25` at `−0.022657`
  (range `−0.287 … −0.023`). The multi-stage comparator wins because it buys
  8-seed breadth cheaply by pruning early, which the fixed budget cannot buy at a
  single checkpoint.
- Out-of-fold selection (fit on 160 prompts, replay on 40 held-out prompts): mean
  held-out `Delta IR` was `−0.029407`, **0 of 5 folds positive**, mean schedule-rank
  Spearman `0.781`, and the selector was coherent (it chose the same neighborhood
  in every fold: `7→3@16`). The gate requires positive mean `Delta IR`, at least
  four positive folds, and rank correlation ≈ `0.8`; it failed on the first two
  and marginally on the third.
- Because the out-of-fold gate failed, the freeze gate was not evaluated and no
  `FROZEN_SCHEDULE.json` was produced. This is a reported, not tuned-away,
  negative result.

Artifacts:

- [`exps/dense_calibration_200/CALIBRATION_REPORT.md`](exps/dense_calibration_200/CALIBRATION_REPORT.md)
- [`exps/dense_calibration_200/replay/oof_gate.json`](exps/dense_calibration_200/replay/oof_gate.json)
- [`exps/dense_calibration_200/replay/oof_folds.csv`](exps/dense_calibration_200/replay/oof_folds.csv)
- [`exps/dense_calibration_200/replay/cell_estimates.csv`](exps/dense_calibration_200/replay/cell_estimates.csv)

The narrow supported claim is that, on this calibration corpus and seed pool, the
pre-registered single-stage search based on shared `p_miss`/`ell` surfaces did not
find any policy expected to beat progressive PSP on ImageReward under the fixed
`M*t + K*(64−t) ≤ 256` budget.

## Adaptive-K screen-and-commit result

A separate development cycle (see
[`exps/adaptive_k_sd15/`](exps/adaptive_k_sd15/README.md)) tested whether survivor
compute can be redistributed from easy to ambiguous prompts at the same aggregate
budget (`sum_p K_p = 2N`, exactly 256 average logical UNet evaluations). The
method fixes `M=10`, `t=16`, `K in {1,2,3}`, predicts realized regrets from
step-16 score geometry, and allocates `K` with an exact dynamic program.

The out-of-fold gate **FAILED**, so no GenEval-553 validation was run.

- Adaptive-K **beats fixed `10->2`** out of fold: mean `Delta IR = +0.004082`,
  positive in 4 of 5 held-out folds.
- Adaptive-K **loses to PSP** out of fold: mean `Delta IR = -0.047917`, 0 of 5
  folds positive. A single prune-and-commit decision at step 16 cannot recover
  PSP's multi-stage `8->4@16->2@32` advantage.

Report: [`exps/adaptive_k_sd15/calibration/CALIBRATION_REPORT.md`](exps/adaptive_k_sd15/calibration/CALIBRATION_REPORT.md).

## License and citation

The code retains the upstream MIT license. Please cite the PSP paper and clearly
identify any result produced by this TQK experimental extension rather than the
upstream repository.
