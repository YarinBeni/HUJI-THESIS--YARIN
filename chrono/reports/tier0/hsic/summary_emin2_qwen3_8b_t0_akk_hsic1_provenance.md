# E-MIN summary — `emin2_qwen3_8b_t0_akk_hsic1_provenance`

Seeds: 0, 1, 2 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.342 ± 0.004 | +0.434 ± 0.013 | 3 |
| `mask_ruler` | +0.388 ± 0.011 | +0.491 ± 0.018 | 3 |
| `strip_formula` | +0.333 ± 0.016 | +0.427 ± 0.023 | 3 |
| `mask_ruler,strip_formula` | +0.386 ± 0.021 | +0.490 ± 0.025 | 3 |
| `mask_ruler,crop16` | +0.306 ± 0.019 | +0.399 ± 0.022 | 3 |
| `mask_ruler,crop32` | +0.338 ± 0.013 | +0.442 ± 0.028 | 3 |
| `orthonorm` | +0.415 ± 0.009 | +0.504 ± 0.026 | 3 |
| `mask_ruler,drop_span` | +0.382 ± 0.021 | +0.499 ± 0.022 | 3 |

Ruler-block null on `orig` (all seeds × draws): -0.009 ± 0.218, 95% band [-0.422, +0.399] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.002 ± 0.015, 95% band [-0.030, +0.034] (200 ruler permutations, seed 0 scores)

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
