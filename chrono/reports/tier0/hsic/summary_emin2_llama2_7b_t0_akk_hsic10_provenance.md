# E-MIN summary — `emin2_llama2_7b_t0_akk_hsic10_provenance`

Seeds: 0, 1, 2 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.430 ± 0.031 | +0.528 ± 0.043 | 3 |
| `mask_ruler` | +0.462 ± 0.013 | +0.551 ± 0.028 | 3 |
| `strip_formula` | +0.430 ± 0.028 | +0.525 ± 0.042 | 3 |
| `mask_ruler,strip_formula` | +0.458 ± 0.017 | +0.552 ± 0.032 | 3 |
| `mask_ruler,crop16` | +0.384 ± 0.015 | +0.461 ± 0.023 | 3 |
| `mask_ruler,crop32` | +0.419 ± 0.021 | +0.496 ± 0.034 | 3 |
| `orthonorm` | +0.431 ± 0.045 | +0.521 ± 0.051 | 3 |
| `mask_ruler,drop_span` | +0.454 ± 0.016 | +0.542 ± 0.033 | 3 |

Ruler-block null on `orig` (all seeds × draws): -0.008 ± 0.237, 95% band [-0.450, +0.433] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.003 ± 0.018, 95% band [-0.027, +0.037] (200 ruler permutations, seed 0 scores)

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
