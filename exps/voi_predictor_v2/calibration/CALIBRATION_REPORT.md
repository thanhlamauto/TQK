# Predictor v2 calibration report (F0-F3)

## Outcome

Best family/model `F3:logistic`: Q_policy = 0.810968,
Delta vs PSP = -0.041477, headroom capture = 0.5%,
PSP folds = 0/5. Gate **FAIL**.

Weighted classification of `z=1[V>0]` with weight `min(|V|, q95)`, threshold chosen
on training folds. F0 scalar step-16; F1 temporal reward dynamics (12/14/16);
F2 predicted-clean latent drift 14->16; F3 = F1+F2+ImageReward hidden PCA(16).

## Baselines (prompt-level)

| Q_A | Q_B | Q_PSP | Q_oracle |
|---:|---:|---:|---:|
| 0.802471 | 0.810636 | 0.852444 | 0.878000 |

PSP needs capture >= 0.621 of the Oracle-B headroom.

## Families

| Family | Model | Q_policy | Delta vs B | Delta vs PSP | Headroom captured | PSP folds | AUC | % choose B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| F0 | logistic | 0.810676 | +0.000040 | -0.041769 | 0.1% | 0/5 | 0.581 | 0.628 |
| F0 | boost | 0.805863 | -0.004773 | -0.046581 | -7.1% | 0/5 | 0.532 | 0.525 |
| F1 | logistic | 0.808496 | -0.002140 | -0.043948 | -3.2% | 0/5 | 0.566 | 0.567 |
| F1 | boost | 0.807300 | -0.003336 | -0.045144 | -5.0% | 0/5 | 0.534 | 0.475 |
| F2 | logistic | 0.806805 | -0.003831 | -0.045639 | -5.7% | 0/5 | 0.556 | 0.565 |
| F2 | boost | 0.806509 | -0.004127 | -0.045935 | -6.1% | 0/5 | 0.526 | 0.447 |
| F3 | logistic | 0.810968 | +0.000331 | -0.041477 | 0.5% | 0/5 | 0.557 | 0.533 |
| F3 | boost | 0.810655 | +0.000019 | -0.041789 | 0.0% | 0/5 | 0.542 | 0.490 |

## High-value diagnostics (best model)

| Group | caught | precision | recall |
|---|---:|---:|---:|
| top10pct | 2343.0 | 0.0219595861138187 | 0.5788043478260869 |
| top25pct | 6122.0 | 0.05737797105795906 | 0.6049407114624505 |
| top50pct | 11964.0 | 0.11213166379245708 | 0.5910775159330073 |
| mass | nan | nan | nan |

## Gate

| Condition | Value | Pass |
|---|---:|---|
| Delta vs PSP > 0 | -0.041477 | False |
| >= 4/5 folds non-negative | 0/5 | False |
| capture > 65% | 0.5% | False |
| **Overall** |  | **False** |
