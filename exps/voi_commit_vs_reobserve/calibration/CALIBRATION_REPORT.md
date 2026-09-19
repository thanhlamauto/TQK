# Value-of-Information calibration report

## Outcome

Oracle headroom exists. Learned-policy gate: **FAIL** (Adaptive - PSP = -0.046711, beats A: True, beats B: False, positive PSP folds 0/5).

Two equal-compute actions at step 16: A = 10->2@16->final (256 NFE) and
B = 10->4@16->1@32->final (256 NFE). PSP = 8->4@16->2@32 (256 NFE) uses the first
8 of the ordered pool. Label `V = Y_B - Y_A`; features are step-16 score geometry
only. Pools are prompt-grouped and metrics aggregate within prompt before
averaging, so hundreds of pools per prompt do not fake sample size.

## Oracle headroom (computed before any training)

| Fold | Oracle | A | B | PSP | Oracle-A | Oracle-B | Oracle-PSP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.949829 | 0.862128 | 0.888481 | 0.932323 | +0.087700 | +0.061347 | +0.017506 |
| 1 | 0.887753 | 0.820512 | 0.795609 | 0.867727 | +0.067241 | +0.092144 | +0.020026 |
| 2 | 0.867390 | 0.790527 | 0.804671 | 0.827289 | +0.076863 | +0.062718 | +0.040101 |
| 3 | 0.996139 | 0.938840 | 0.942395 | 0.966478 | +0.057299 | +0.053744 | +0.029661 |
| 4 | 0.688890 | 0.600349 | 0.622024 | 0.668405 | +0.088540 | +0.066865 | +0.020485 |
| overall | 0.878000 | 0.802471 | 0.810636 | 0.852444 | +0.075529 | +0.067364 | +0.025556 |

Headroom over PSP is the mandatory first gate. Oracle-A measures A/B
heterogeneity; Oracle-B measures how much A adds on top of B.

## Selector metrics and OOF policy

| Model | MAE | RMSE | Spearman | sign acc | precision V>0 | recall V>0 | Delta vs PSP | Delta vs A | Delta vs B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ambiguity_g2 | 0.1482 | 0.2933 | -0.049 | 0.205 | 0.201 | 0.987 | -0.042113 | +0.007860 | -0.000305 |
| ambiguity_min_g2_g4 | 0.1480 | 0.2933 | -0.049 | 0.204 | 0.202 | 0.996 | -0.042043 | +0.007930 | -0.000235 |
| ridge | 0.1547 | 0.2936 | 0.030 | 0.412 | 0.227 | 0.795 | -0.042085 | +0.007888 | -0.000277 |
| boost | 0.1628 | 0.2976 | -0.008 | 0.473 | 0.218 | 0.620 | -0.046711 | +0.003262 | -0.004903 |

## Realized VOI by predicted VOI quintile (primary model)

| Quintile | mean realized V | pools |
|---:|---:|---:|
| 1 | +0.023427 | 40000 |
| 2 | +0.001202 | 40013 |
| 3 | -0.000607 | 39988 |
| 4 | +0.008775 | 39999 |
| 5 | +0.008027 | 40000 |

## Ambiguity analysis (primary model)

| Chosen | mean g2 | mean g4 | mean top4_std | mean std | mean V |
|---|---:|---:|---:|---:|---:|
| A | 4.5357 | 1.3414 | 0.4425 | 0.6400 | +0.011558 |
| B | 2.4686 | 1.1948 | 0.2490 | 0.5352 | +0.005665 |


## Interpretation guardrails

This is calibration evidence only. HPS and official GenEval were never inspected
and never influenced the selector. If the gate failed, no parameter was tuned in
response and a broader action family requires a new development cycle.
