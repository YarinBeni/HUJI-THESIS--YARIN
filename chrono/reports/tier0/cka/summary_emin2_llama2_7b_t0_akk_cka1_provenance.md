# E-MIN summary — `emin2_llama2_7b_t0_akk_cka1_provenance`

Seeds: 0, 1, 2 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.420 ± 0.023 | +0.511 ± 0.033 | 3 |
| `mask_ruler` | +0.455 ± 0.019 | +0.540 ± 0.021 | 3 |
| `strip_formula` | +0.421 ± 0.021 | +0.510 ± 0.032 | 3 |
| `mask_ruler,strip_formula` | +0.452 ± 0.022 | +0.540 ± 0.024 | 3 |
| `mask_ruler,crop16` | +0.377 ± 0.027 | +0.452 ± 0.021 | 3 |
| `mask_ruler,crop32` | +0.411 ± 0.030 | +0.486 ± 0.027 | 3 |
| `orthonorm` | +0.421 ± 0.032 | +0.506 ± 0.037 | 3 |
| `mask_ruler,drop_span` | +0.448 ± 0.024 | +0.530 ± 0.026 | 3 |

Ruler-block null on `orig` (all seeds × draws): -0.007 ± 0.233, 95% band [-0.439, +0.415] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.003 ± 0.017, 95% band [-0.026, +0.036] (200 ruler permutations, seed 0 scores)

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
