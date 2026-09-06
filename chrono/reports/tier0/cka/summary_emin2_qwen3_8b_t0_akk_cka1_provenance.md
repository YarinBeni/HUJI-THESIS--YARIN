# E-MIN summary — `emin2_qwen3_8b_t0_akk_cka1_provenance`

Seeds: 0, 1, 2 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.332 ± 0.034 | +0.430 ± 0.038 | 3 |
| `mask_ruler` | +0.374 ± 0.037 | +0.477 ± 0.038 | 3 |
| `strip_formula` | +0.326 ± 0.032 | +0.424 ± 0.037 | 3 |
| `mask_ruler,strip_formula` | +0.375 ± 0.035 | +0.474 ± 0.037 | 3 |
| `mask_ruler,crop16` | +0.297 ± 0.012 | +0.378 ± 0.023 | 3 |
| `mask_ruler,crop32` | +0.329 ± 0.024 | +0.423 ± 0.031 | 3 |
| `orthonorm` | +0.406 ± 0.029 | +0.507 ± 0.027 | 3 |
| `mask_ruler,drop_span` | +0.367 ± 0.042 | +0.478 ± 0.044 | 3 |

Ruler-block null on `orig` (all seeds × draws): -0.008 ± 0.222, 95% band [-0.434, +0.401] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.003 ± 0.015, 95% band [-0.028, +0.032] (200 ruler permutations, seed 0 scores)

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
