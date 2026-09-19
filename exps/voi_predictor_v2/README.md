# Predictor v2: weighted policy classification with temporal/dynamics features

Same two equal-compute actions as the VOI experiment (A = `10->2@16`, B =
`10->4@16->1@32`, both 256 NFE).  The predictor is changed in two ways:

1. **Objective**: weighted classification of `z = 1[V>0]` with weight `|V|`
   (capped at its 95th percentile), which is Bayes-aligned with final
   ImageReward, instead of MSE regression of `V`.
2. **Information**: four pre-registered feature families F0-F3 that add temporal
   reward dynamics (steps 12/14), predicted-clean latent drift, and ImageReward
   hidden features.

The key metric is oracle-headroom capture
`H = (Q_policy - max(Q_A,Q_B)) / (Q_oracle - max(Q_A,Q_B))`; from the previous
calibration PSP requires `H >= 62%`, and the gate uses `H > 65%`.

- `PROTOCOL.md` - frozen protocol.
- `prepare_bank.py` - compact bank (steps 12/14/16/32/64) from the independent corpus.
- `simulate_pools.py` - ordered pools, A/B/PSP replay, F0/F1 features.
- `voi_v2_models.py` - feature builders, weighted classifiers, H metric.
- `run_cv_v2.py` - grouped OOF, threshold selection, diagnostics, gate.
- `test_voi_v2.py` - CPU-only invariants.
- `run_calibration.sh` - F0/F1 end-to-end.

```bash
python test_voi_v2.py
bash run_calibration.sh     # F0/F1 (no GPU)
bash run_dynamics.sh        # F2/F3 dynamics bank + all four families
```

## Result

Both the weighted-policy objective and the extra information failed to capture
the oracle headroom. PSP requires capture >= 62% of the `Oracle - B` headroom
(`(0.852444-0.810636)/(0.878000-0.810636) = 0.621`).

| Family | best model | Q_policy | Delta vs PSP | Headroom captured | PSP folds | AUC |
|---|---|---:|---:|---:|---:|---:|
| F0 scalar step-16 | logistic | 0.810676 | -0.041769 | 0.1% | 0/5 | 0.581 |
| F1 temporal reward | logistic | 0.808496 | -0.043948 | -3.2% | 0/5 | 0.566 |
| F2 latent drift | logistic | 0.806805 | -0.045639 | -5.7% | 0/5 | 0.556 |
| F3 F1+F2+IR hidden | logistic | 0.810968 | -0.041477 | 0.5% | 0/5 | 0.557 |

- No family comes close to the 62% capture needed to match PSP, and none beats
  PSP (0/5 folds, AUC barely above chance).
- Temporal reward dynamics (F1), predicted-clean latent drift (F2) and frozen
  768-d ImageReward hidden features (F3) add essentially **no** usable signal.
- The high-value diagnostic shows the policy captures ~60% of positive reward
  mass but incurs almost as much negative mass (net ~+1700 on ~15000), i.e. it
  picks B too broadly and not selectively.

**Bottleneck: information, not predictor objective or capacity.** The decision
state before step 16, as measured by scalar rewards, their dynamics, latent drift
and ImageReward hidden features, does not determine whether re-observation is
worth it. Per the pre-registered stop rule, no GenEval-553 validation was run;
the natural next step is a noise-aware verifier (e.g. TTSnap), not a more complex
predictor.

See `calibration/CALIBRATION_REPORT.md`, `family_results_dynamics.csv`,
`gate_dynamics.json`, `high_value_diagnostics_dynamics.csv`.
