# E-MIN summary — `emin2_llama2_7b_t0_akk_cka5_provenance`

Seeds: 0, 1, 2 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.422 ± 0.026 | +0.505 ± 0.035 | 3 |
| `mask_ruler` | +0.453 ± 0.023 | +0.532 ± 0.025 | 3 |
| `strip_formula` | +0.422 ± 0.025 | +0.505 ± 0.034 | 3 |
| `mask_ruler,strip_formula` | +0.452 ± 0.027 | +0.535 ± 0.028 | 3 |
| `mask_ruler,crop16` | +0.377 ± 0.026 | +0.445 ± 0.018 | 3 |
| `mask_ruler,crop32` | +0.413 ± 0.030 | +0.480 ± 0.023 | 3 |
| `orthonorm` | +0.421 ± 0.039 | +0.507 ± 0.046 | 3 |
| `mask_ruler,drop_span` | +0.447 ± 0.026 | +0.522 ± 0.028 | 3 |

Ruler-block null on `orig` (all seeds × draws): -0.007 ± 0.233, 95% band [-0.443, +0.421] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.003 ± 0.018, 95% band [-0.026, +0.036] (200 ruler permutations, seed 0 scores)

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
