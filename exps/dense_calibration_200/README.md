# Dense single-stage calibration (pre-registered next experiment)

Calibrates exactly one single-stage pruning policy for deterministic SD1.5/DDIM
inference under a logical budget of at most 256 UNet evaluations, using a denser
search than the completed `single_stage_calibration` experiment.

Differences from the completed calibration:

* 200 independent ImageReward prompts (new selection seed `20260920`) instead of
  120;
* the full search grid `t in {10..28}` x `K in {1,2,3}` = 57 policies instead of
  13 checkpoints;
* ImageReward stored at every searched step (`X[p,i,t]`, `t=10..28`) instead of a
  13-point subset;
* shared reliability surfaces for `p_miss` and `ell` instead of independent cell
  means;
* deterministic five-fold out-of-fold selection and an 80% lower-confidence
  freeze gate.

See `PROTOCOL.md` for the frozen constants and `reliability_models.py` for the
exact model formulas.  `run_all.sh` runs preflight, the 200x25 trajectory bank,
the OOF/LCB80 analysis, and — only if the freeze gate passes — the confirmatory
553-prompt experiment.

CPU-only correctness tests run before any GPU work:

```
python test_analysis.py
```
