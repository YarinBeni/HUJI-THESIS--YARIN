# S3 — does SSL pretraining help dating the 1,193 dated inscriptions?

Read-out: SLA §7 on centred out-of-fold scores, `orig` condition. `gkf` is the POOLED Spearman over the held-out docs of the ruler-grouped folds (per-fold rho is undefined there: 39 of 40 rulers carry a single year); `mc` is the mean over the frozen balanced draws. Mean +- sd over seeds.

**The SSL claim is a difference.** Compare each `barlow`/`jepa` row with the `none` row of the same encoder and label fraction; the absolute rho is dominated by the frozen encoder underneath.

| encoder | init | label frac | seeds | folds | mc rho | gkf rho (pooled) |
|---|---|---|---|---|---|---|
| `cunei400m` | barlow | 25% | 3 | 5 | 0.536 ± 0.085 | 0.459 ± 0.058 |
| `cunei400m` | barlow | 50% | 3 | 5 | 0.606 ± 0.017 | 0.534 ± 0.018 |
| `cunei400m` | barlow | 100% | 3 | 5 | 0.590 ± 0.020 | 0.502 ± 0.019 |
| `cunei400m` | jepa | 25% | 3 | 5 | 0.534 ± 0.040 | 0.422 ± 0.020 |
| `cunei400m` | jepa | 50% | 3 | 5 | 0.622 ± 0.023 | 0.508 ± 0.019 |
| `cunei400m` | jepa | 100% | 3 | 5 | 0.625 ± 0.018 | 0.506 ± 0.024 |
| `cunei400m` | none | 25% | 3 | 5 | 0.534 ± 0.024 | 0.394 ± 0.020 |
| `cunei400m` | none | 50% | 3 | 5 | 0.663 ± 0.011 | 0.537 ± 0.007 |
| `cunei400m` | none | 100% | 3 | 5 | 0.603 ± 0.013 | 0.508 ± 0.016 |
| `llama2_7b` | barlow | 25% | 1 | 5 | 0.547 | 0.440 |
| `llama2_7b` | barlow | 50% | 2 | 5 | 0.551 ± 0.046 | 0.491 ± 0.001 |

## SSL init minus the `none` control (same encoder, same label fraction)

| encoder | init | label frac | Δ mc rho | Δ gkf rho |
|---|---|---|---|---|
| `cunei400m` | barlow | 25% | +0.001 | +0.066 |
| `cunei400m` | barlow | 50% | -0.057 | -0.003 |
| `cunei400m` | barlow | 100% | -0.013 | -0.006 |
| `cunei400m` | jepa | 25% | +0.000 | +0.028 |
| `cunei400m` | jepa | 50% | -0.042 | -0.029 |
| `cunei400m` | jepa | 100% | +0.023 | -0.002 |
