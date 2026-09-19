# Adaptive-K Screen-and-Commit (SD1.5)

Tests whether survivor compute can be redistributed from easy prompts to
ambiguous prompts, at the same aggregate budget (exactly 256 logical UNet
evaluations per prompt on average), to beat progressive PSP.

The method is intentionally simple and fixed: 10 initial seeds, prune at step 16,
choose `K in {1,2,3}`, finish those K candidates, pick the final winner by final
ImageReward.

- `PROTOCOL.md` - frozen protocol, features, predictors, exact-budget allocator,
  prompt-grouped CV and OOF gate.
- `prepare_bank.py` - builds the compact calibration bank from the independent
  non-GenEval ImageReward corpus.
- `simulate_subsets.py` - 1000 offline 10-seed pools per prompt (RNG seed
  20260919), labels and PSP replay.
- `regret_models.py` - features, Model A (isotonic), Model B (boosted), monotone
  projection, exact dynamic-programming allocation.
- `run_cv.py` - prompt-grouped OOF evaluation, baselines, gate, diagnostics,
  report and plots.
- `test_adaptive_k.py` - CPU-only invariant tests (run before any GPU work).
- `run_calibration.sh` - end-to-end calibration pipeline.

Calibration output lives under `calibration/`; if the OOF gate passes, the frozen
policy is written under `policy/` and the 553-prompt validation under
`validation/` (added by a separate implementation after the gate).

```bash
python test_adaptive_k.py
bash run_calibration.sh
```

## Result

The out-of-fold gate **FAILED**, so no predictor was frozen and no GenEval
validation was run (pre-registered stop rule).

- Adaptive-K **does** add value over the already-tested fixed `10->2` schedule:
  mean OOF `Delta IR` = `+0.004082`, positive in 4 of 5 folds (condition B and C
  pass).
- Adaptive-K **does not** beat PSP: mean OOF `Delta IR` = `-0.047917`, 0 of 5
  folds positive (condition A fails). A single prune-and-commit decision at
  step 16 cannot recover PSP's multi-stage (`8->4@16->2@32`) advantage.
- The regret predictor generalizes (held-out Spearman ~0.26-0.34 for the boosted
  model versus ~0.05 for the isotonic baseline), and the exact-budget allocator
  kept `sum K_p = 2N` with `n1 = n3` in every batch.
- Most useful features are the absolute score spread (`max-min`, `top3_mean`,
  `top3_std`, `std`), not the local boundary gaps (`g1..g3`).

See `calibration/CALIBRATION_REPORT.md` and `calibration/oof_gate.json`.
