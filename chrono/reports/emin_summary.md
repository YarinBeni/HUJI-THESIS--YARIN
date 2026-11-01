# E-MIN summary — `emin_thalesian`

Seeds: 0, 1, 2, 3, 4 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.317 ± 0.025 | +0.398 ± 0.022 | 5 |
| `mask_ruler` | +0.285 ± 0.028 | +0.370 ± 0.018 | 5 |
| `strip_formula` | +0.314 ± 0.025 | +0.398 ± 0.023 | 5 |
| `mask_ruler,strip_formula` | +0.283 ± 0.028 | +0.372 ± 0.020 | 5 |
| `mask_ruler,crop16` | +0.253 ± 0.019 | +0.320 ± 0.013 | 5 |
| `mask_ruler,crop32` | +0.276 ± 0.024 | +0.359 ± 0.014 | 5 |
| `orthonorm` | +0.334 ± 0.025 | +0.398 ± 0.024 | 5 |
| `mask_ruler,drop_span` | +0.296 ± 0.028 | +0.373 ± 0.020 | 5 |

Ruler-block null on `orig` (all seeds × draws): -0.007 ± 0.194, 95% band [-0.370, +0.370] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.002 ± 0.013, 95% band [-0.020, +0.028] (200 ruler permutations, seed 0 scores)

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
