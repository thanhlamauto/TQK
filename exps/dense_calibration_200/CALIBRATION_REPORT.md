# Dense single-stage calibration report (README sections 3-6)

## Outcome

Out-of-fold gate **FAILED**; per the pre-registered stop rule the confirmatory 553-prompt phase was not run.

* Calibration corpus: 200 fixed prompts from the repository ImageReward
  `test_ir.json`, selection seed 20260920, overlap with the previous 120-prompt calibration = 56.
  Zero exact overlap with the 553 GenEval prompts.
* Search space: exactly 57 policies ``t in {10..28}``, ``K in {1,2,3}``, with
  ``M(t,K)=floor((256-K*(64-t))/t)``. ``M`` was never tuned independently.
* Bank: 200 prompts x 25 independent trajectories x
  20 checkpoints (10..28 plus 32 for the fixed PSP replay) plus
  the final step, scored with ImageReward on the scheduler's own
  ``pred_original_sample``.

## Section 3 - breadth value and pruning regret

* Invariant ``max |mean regret - p_miss * ell|`` = 5.551e-17.
* Best empirical cell (all 200 prompts): **7→2@25** with
  Delta IR -0.022657.
* Fixed PSP reference: ``8→4@16→2@32`` = 256 logical UNet
  evaluations; mean calibration IR 0.851613.

## Section 4 - shared reliability surfaces

* ``p_miss``: logistic with hinge basis in ``q=t/64`` and ``w_j >= 0`` (monotone
  non-increasing in denoising depth), plus ``log(M)`` and ``K/M``; L2 = 1e-3.
* ``ell``: Gamma regression with log link on ``[1, log(M), K/M, q, q^2]``; L2 = 1e-3.
* Full-200 fitted surfaces produced the bootstrap table below.

## Section 5 - five-fold out-of-fold selection

| fold | selected | held Delta IR | positive | Spearman | predicted winner rank | predicted winner |
|---:|---|---:|:--:|---:|---:|---|
| 0 | 7→3@16 | -0.023111 | no | 0.815 | 7 | 7→3@16 |
| 1 | 7→3@16 | -0.023476 | no | 0.813 | 2 | 6→3@21 |
| 2 | 7→3@16 | -0.043729 | no | 0.790 | 12 | 8→3@12 |
| 3 | 7→3@16 | -0.022738 | no | 0.654 | 7 | 6→3@21 |
| 4 | 7→3@16 | -0.033981 | no | 0.833 | 3 | 7→3@16 |

| gate | value | threshold | pass |
|---|---:|---:|:--:|
| mean held Delta IR > 0 | -0.029407 | > 0 | False |
| positive folds | 0 | >= 4 | False |
| mean rank Spearman | 0.781 | >= 0.8 | False |
| coherent neighborhood | span t=0, span K=0 | t<=4, K<=1 | True |
| **overall OOF gate** |  |  | **False** |

Predicted winner was in the actual top-3 in 2/5 folds and
top-5 in 2/5 folds.

## Section 6 - freeze gate (prompt bootstrap, 1000 resamples)

Top policies by 80% lower confidence bound of predicted Delta IR:

| policy | predicted Delta IR | LCB80 | CI95 low | CI95 high |
|---|---:|---:|---:|---:|


## Interpretation guardrails

* This is calibration evidence only. HPS and official GenEval were not inspected
  during calibration and selection.
* The confirmatory endpoint (README section 7) is a separate 553-prompt, three
  fresh-seed comparison and is run only if this freeze gate passes.
