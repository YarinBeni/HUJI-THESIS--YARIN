# E-MIN summary — `emin2_cunei400m_t0_akk_cka1_provenance`

Seeds: 0, 1, 2 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.468 ± 0.021 | +0.576 ± 0.016 | 3 |
| `mask_ruler` | +0.430 ± 0.010 | +0.521 ± 0.014 | 3 |
| `strip_formula` | +0.439 ± 0.019 | +0.549 ± 0.015 | 3 |
| `mask_ruler,strip_formula` | +0.401 ± 0.012 | +0.496 ± 0.017 | 3 |
| `mask_ruler,crop16` | +0.384 ± 0.007 | +0.453 ± 0.007 | 3 |
| `mask_ruler,crop32` | +0.390 ± 0.009 | +0.486 ± 0.010 | 3 |
| `orthonorm` | +0.438 ± 0.033 | +0.549 ± 0.025 | 3 |
| `mask_ruler,drop_span` | +0.417 ± 0.013 | +0.506 ± 0.017 | 3 |

Ruler-block null on `orig` (all seeds × draws): -0.010 ± 0.252, 95% band [-0.485, +0.467] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.002 ± 0.017, 95% band [-0.028, +0.033] (200 ruler permutations, seed 0 scores)

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
