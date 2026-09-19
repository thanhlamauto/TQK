# Value-of-Information: commit vs re-observe (SD1.5)

Given the 10 step-16 ImageReward scores, decide between two exactly-equal-compute
actions:

* **A** - commit now: `10 -> 2 @16 -> final` (256 NFE)
* **B** - buy information: `10 -> 4 @16 -> 1 @32 -> final` (256 NFE)

Baseline PSP: `8 -> 4 @16 -> 2 @32 -> final` (256 NFE) on the first 8 candidates.
Label `V = Y_B - Y_A`; oracle `max(Y_A, Y_B)`.

The oracle-headroom gate is evaluated **before any predictor is trained**: if the
`{A,B}` action family cannot beat PSP even with a perfect selector, the experiment
stops and reports action-family insufficiency.

- `PROTOCOL.md` - frozen protocol.
- `prepare_bank.py` - compact calibration bank from the independent corpus.
- `simulate_pools.py` - ordered 10-seed pools (seed 20260920), A/B/PSP replay, V.
- `voi_models.py` - features, low-capacity models, NFE accounting.
- `run_cv.py` - oracle gate, grouped OOF selector, learned-policy gate, reports.
- `test_voi.py` - CPU-only invariants (run first).
- `run_calibration.sh` - end-to-end calibration.

```bash
python test_voi.py
bash run_calibration.sh
```

## Result

The **oracle gate passed but the learned-policy gate failed**, so no selector was
frozen and no GenEval-553 validation was launched (pre-registered stop rule).

Oracle headroom (prompt-level, no training):

| Quantity | Value |
|---|---:|
| Oracle | 0.878000 |
| Fixed A | 0.802471 |
| Fixed B | 0.810636 |
| PSP | 0.852444 |
| **Oracle - PSP** | **+0.025556** (positive in 5/5 folds) |
| Oracle - A | +0.075529 |
| Oracle - B | +0.067364 |

So the `{A,B}` action family **does** contain enough headroom to beat PSP: a
perfect per-pool selector would win by ~0.026 IR. Action heterogeneity is real.

Learned selector (primary `HistGradientBoostingRegressor`):

- MAE 0.1628, RMSE 0.2976, **Spearman(V_hat, V) = -0.008**, sign accuracy 0.473.
  All simple models are near-uninformative about `V` (Spearman -0.05..+0.03).
- Adaptive OOF: beats Fixed A (+0.0033) but not Fixed B (-0.0049) and loses to PSP
  (-0.0467); 0/5 folds positive vs PSP.
- Realized `V` by predicted-`V` quintile is **not monotone**, so predicted VOI does
  not calibrate to realized VOI.

**Bottleneck: predictor quality, not action-family expressiveness.** Step-16 score
geometry (gaps, spread, z-scores) does not predict whether re-observing at step 32
will help. This is a reported negative result; improving the predictor is a new
experiment.

See `calibration/CALIBRATION_REPORT.md`, `calibration/oracle_headroom.csv`,
`calibration/regret_metrics.csv`, `calibration/fold_results.csv`.
