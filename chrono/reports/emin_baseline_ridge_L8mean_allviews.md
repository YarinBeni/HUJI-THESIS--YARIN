# E-MIN summary — `baseline_ridge_L8mean_allviews`

Seeds: 0 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.271 ± nan | +0.377 ± nan | 1 |
| `mask_ruler` | +0.249 ± nan | +0.345 ± nan | 1 |
| `strip_formula` | +0.268 ± nan | +0.377 ± nan | 1 |
| `mask_ruler,strip_formula` | +0.246 ± nan | +0.346 ± nan | 1 |
| `mask_ruler,crop16` | +0.208 ± nan | +0.277 ± nan | 1 |
| `mask_ruler,crop32` | +0.227 ± nan | +0.318 ± nan | 1 |
| `orthonorm` | +0.264 ± nan | +0.394 ± nan | 1 |
| `mask_ruler,drop_span` | +0.252 ± nan | +0.340 ± nan | 1 |

Ruler-block null on `orig` (all seeds × draws): -0.001 ± 0.174, 95% band [-0.375, +0.299] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.002 ± 0.012, 95% band [-0.021, +0.025] (200 ruler permutations, seed 0 scores)

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
