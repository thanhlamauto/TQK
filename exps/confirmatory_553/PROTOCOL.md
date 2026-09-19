# Confirmatory protocol (README section 7, frozen before 553 inference)

Runs only after the dense-calibration freeze gate has produced a
`FROZEN_SCHEDULE.json` with `LCB80(Delta IR) > 0`.

## Fixed comparison

- Frozen single-stage schedule read from
  `../dense_calibration_200/FROZEN_SCHEDULE.json` (M, K, checkpoint, budget).
- Fixed comparator PSP `8->4@16->2@32->1@64` = 256 logical UNet evaluations.
- Stable Diffusion 1.5, deterministic DDIM, 64 steps, `eta=0`, fp16, guidance 7.5.
- All 553 official GenEval prompts, untouched during calibration.

## Fresh seeds

- Three pre-recorded candidate-pool base seeds: `20260921, 20260922, 20260923`,
  disjoint from calibration (`20260920`) and the previous validation run
  (`20260919`, and the original bank's implicit base 0).
- Candidate seed formula `base + prompt_id*32 + candidate_id`.
- For each prompt and repetition both methods start from the identical candidate
  pool, so every prompt-level difference is paired.

## Endpoints and inference

- Primary endpoint: ImageReward.
- Secondary endpoints: HPS v2.1 and official GenEval.
- For prompt `p` and repetition `r` compute `d[p,r]`; average the three
  repetitions within prompt first, then bootstrap the 553 prompt-level averages
  10,000 times (seed `20260920`).  The 1,659 images are not treated as
  independent samples.
- Strong success requires positive mean `Delta IR` and a 95% paired-confidence
  lower bound above zero.  HPS and GenEval must not show clear degradation; if
  ImageReward wins while an independent metric declines, the claim is restricted
  to better optimization of ImageReward.

## Reporting

Always report logical UNet evaluations, runtime mean/median/p90, throughput per
GPU, peak allocated VRAM, batched verifier calls, candidate scores per prompt,
all prompt IDs and seeds, and all paired confidence intervals.
