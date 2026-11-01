# E-MIN summary — `emin2_qwen3_8b_t0_akk_cka5_provenance`

Seeds: 0, 1, 2 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.352 ± 0.038 | +0.434 ± 0.024 | 3 |
| `mask_ruler` | +0.388 ± 0.039 | +0.479 ± 0.024 | 3 |
| `strip_formula` | +0.347 ± 0.038 | +0.432 ± 0.025 | 3 |
| `mask_ruler,strip_formula` | +0.390 ± 0.035 | +0.480 ± 0.023 | 3 |
| `mask_ruler,crop16` | +0.305 ± 0.021 | +0.382 ± 0.020 | 3 |
| `mask_ruler,crop32` | +0.340 ± 0.033 | +0.433 ± 0.025 | 3 |
| `orthonorm` | +0.413 ± 0.051 | +0.503 ± 0.036 | 3 |
| `mask_ruler,drop_span` | +0.383 ± 0.042 | +0.481 ± 0.024 | 3 |

Ruler-block null on `orig` (all seeds × draws): -0.009 ± 0.219, 95% band [-0.424, +0.391] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.002 ± 0.015, 95% band [-0.031, +0.032] (200 ruler permutations, seed 0 scores)

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
