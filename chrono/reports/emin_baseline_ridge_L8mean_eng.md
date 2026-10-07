# E-MIN summary — `baseline_ridge_L8mean_eng`

Seeds: 0 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.264 ± nan | +0.378 ± nan | 1 |
| `mask_ruler` | +0.216 ± nan | +0.327 ± nan | 1 |
| `strip_formula` | +0.264 ± nan | +0.378 ± nan | 1 |
| `mask_ruler,strip_formula` | +0.216 ± nan | +0.327 ± nan | 1 |
| `mask_ruler,crop16` | +0.112 ± nan | +0.179 ± nan | 1 |
| `mask_ruler,crop32` | +0.168 ± nan | +0.273 ± nan | 1 |
| `orthonorm` | +0.322 ± nan | +0.402 ± nan | 1 |
| `mask_ruler,drop_span` | +0.218 ± nan | +0.323 ± nan | 1 |

Ruler-block null on `orig` (all seeds × draws): -0.006 ± 0.176, 95% band [-0.355, +0.322] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.002 ± 0.011, 95% band [-0.020, +0.023] (200 ruler permutations, seed 0 scores)

## C2 baseline at the same features (Thalesian/AKK_300m L8 mean, lang=akk)

| probe | split | ρ |
|---|---|---|
| pls | gkf_ruler | +0.141 |
| pls | mc_balanced | +0.126 |
| pls::rowl2 | gkf_ruler | +0.154 |
| pls::rowl2 | mc_balanced | +0.148 |
| ridge | gkf_ruler | +0.260 |
| ridge | mc_balanced | +0.287 |
| ridge::rowl2 | gkf_ruler | +0.264 |
| ridge::rowl2 | mc_balanced | +0.296 |

Baseline rows are ONE cross-fit (no seeds); the head has 5. Both use the same folds, the same pooled/mc read-out and the same corpus, so the columns are directly comparable.
