# S3 — does SSL pretraining help dating the 1,193 dated inscriptions?

Read-out: SLA §7 on centred out-of-fold scores, `orig` condition. `gkf` is the POOLED Spearman over the held-out docs of the ruler-grouped folds (per-fold rho is undefined there: 39 of 40 rulers carry a single year); `mc` is the mean over the frozen balanced draws. Mean +- sd over seeds.

**The SSL claim is a difference.** Compare each `barlow`/`jepa` row with the `none` row of the same encoder and label fraction; the absolute rho is dominated by the frozen encoder underneath.

| encoder | init | label frac | head width | seeds | folds | mc rho | gkf rho (pooled) |
|---|---|---|---|---|---|---|---|
| `cunei400m` | barlow | 25% | 512 | 3 | 5 | 0.536 ± 0.085 | 0.459 ± 0.058 |
| `cunei400m` | barlow | 50% | 512 | 3 | 5 | 0.606 ± 0.017 | 0.534 ± 0.018 |
| `cunei400m` | barlow | 100% | 512 | 3 | 5 | 0.590 ± 0.020 | 0.502 ± 0.019 |
| `cunei400m` | barlow_wdated | 25% | 512 | 3 | 5 | 0.536 ± 0.085 | 0.460 ± 0.061 |
| `cunei400m` | barlow_wdated | 100% | 512 | 3 | 5 | 0.588 ± 0.018 | 0.504 ± 0.018 |
| `cunei400m` | byol_wdated | 25% | 512 | 3 | 5 | 0.514 ± 0.040 | 0.385 ± 0.051 |
| `cunei400m` | byol_wdated | 100% | 512 | 3 | 5 | 0.607 ± 0.008 | 0.491 ± 0.013 |
| `cunei400m` | jepa | 25% | 512 | 3 | 5 | 0.534 ± 0.040 | 0.422 ± 0.020 |
| `cunei400m` | jepa | 50% | 512 | 3 | 5 | 0.622 ± 0.023 | 0.508 ± 0.019 |
| `cunei400m` | jepa | 100% | 512 | 3 | 5 | 0.625 ± 0.018 | 0.506 ± 0.024 |
| `cunei400m` | none | 25% | 512 | 3 | 5 | 0.534 ± 0.024 | 0.394 ± 0.020 |
| `cunei400m` | none | 50% | 512 | 3 | 5 | 0.663 ± 0.011 | 0.537 ± 0.007 |
| `cunei400m` | none | 100% | 128 | 3 | 5 | 0.638 ± 0.011 | 0.492 ± 0.022 |
| `cunei400m` | none | 100% | 512 | 3 | 5 | 0.603 ± 0.013 | 0.508 ± 0.016 |
| `cunei400m` | none | 100% | 2048 | 3 | 5 | 0.587 ± 0.017 | 0.499 ± 0.023 |
| `llama2_7b` | barlow | 25% | 512 | 3 | 5 | 0.520 ± 0.036 | 0.440 ± 0.024 |
| `llama2_7b` | barlow | 50% | 512 | 3 | 5 | 0.542 ± 0.036 | 0.488 ± 0.005 |
| `llama2_7b` | barlow | 100% | 512 | 3 | 5 | 0.527 ± 0.045 | 0.453 ± 0.037 |
| `llama2_7b` | jepa | 25% | 512 | 3 | 5 | 0.514 ± 0.019 | 0.432 ± 0.010 |
| `llama2_7b` | jepa | 50% | 512 | 3 | 5 | 0.520 ± 0.018 | 0.469 ± 0.008 |
| `llama2_7b` | jepa | 100% | 512 | 3 | 5 | 0.506 ± 0.049 | 0.441 ± 0.033 |
| `llama2_7b` | none | 25% | 512 | 3 | 5 | 0.486 ± 0.040 | 0.414 ± 0.039 |
| `llama2_7b` | none | 50% | 512 | 3 | 5 | 0.535 ± 0.022 | 0.472 ± 0.012 |
| `llama2_7b` | none | 100% | 128 | 3 | 5 | 0.521 ± 0.023 | 0.421 ± 0.034 |
| `llama2_7b` | none | 100% | 512 | 3 | 5 | 0.529 ± 0.041 | 0.431 ± 0.029 |
| `llama2_7b` | none | 100% | 2048 | 3 | 5 | 0.522 ± 0.036 | 0.432 ± 0.016 |

## SSL init minus the `none` control (same encoder, same label fraction)

| encoder | init | label frac | Δ mc rho | Δ gkf rho |
|---|---|---|---|---|
| `cunei400m` | barlow | 25% | +0.001 | +0.066 |
| `cunei400m` | barlow | 50% | -0.057 | -0.003 |
| `cunei400m` | barlow | 100% | -0.013 | -0.006 |
| `cunei400m` | barlow_wdated | 25% | +0.002 | +0.066 |
| `cunei400m` | barlow_wdated | 100% | -0.014 | -0.004 |
| `cunei400m` | byol_wdated | 25% | -0.021 | -0.008 |
| `cunei400m` | byol_wdated | 100% | +0.005 | -0.017 |
| `cunei400m` | jepa | 25% | +0.000 | +0.028 |
| `cunei400m` | jepa | 50% | -0.042 | -0.029 |
| `cunei400m` | jepa | 100% | +0.023 | -0.002 |
| `llama2_7b` | barlow | 25% | +0.035 | +0.026 |
| `llama2_7b` | barlow | 50% | +0.006 | +0.016 |
| `llama2_7b` | barlow | 100% | -0.002 | +0.022 |
| `llama2_7b` | jepa | 25% | +0.028 | +0.019 |
| `llama2_7b` | jepa | 50% | -0.015 | -0.004 |
| `llama2_7b` | jepa | 100% | -0.023 | +0.010 |
