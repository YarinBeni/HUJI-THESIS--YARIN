# E-MIN summary — `emin2_cunei400m_t0_akk_cka5_provenance`

Seeds: 0, 1, 2 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.452 ± 0.018 | +0.537 ± 0.009 | 3 |
| `mask_ruler` | +0.397 ± 0.007 | +0.464 ± 0.013 | 3 |
| `strip_formula` | +0.420 ± 0.016 | +0.505 ± 0.008 | 3 |
| `mask_ruler,strip_formula` | +0.363 ± 0.010 | +0.433 ± 0.014 | 3 |
| `mask_ruler,crop16` | +0.350 ± 0.009 | +0.394 ± 0.006 | 3 |
| `mask_ruler,crop32` | +0.348 ± 0.007 | +0.421 ± 0.012 | 3 |
| `orthonorm` | +0.411 ± 0.029 | +0.493 ± 0.035 | 3 |
| `mask_ruler,drop_span` | +0.385 ± 0.008 | +0.449 ± 0.015 | 3 |

Ruler-block null on `orig` (all seeds × draws): -0.009 ± 0.239, 95% band [-0.456, +0.440] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.002 ± 0.017, 95% band [-0.026, +0.034] (200 ruler permutations, seed 0 scores)

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
