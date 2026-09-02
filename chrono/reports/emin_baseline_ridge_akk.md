# E-MIN summary — `baseline_ridge_L8mean_akk`

Seeds: 0 · folds per seed: [5] · read-out: SLA §7 (gkf pooled over centred OOF scores; mc = mean of per-draw rho)

## Chrono-Barlow head, by condition (mean ± sd across seeds)

| condition | gkf pooled ρ | mc ρ | seeds |
|---|---|---|---|
| `orig` | +0.163 ± nan | +0.094 ± nan | 1 |
| `mask_ruler` | +0.240 ± nan | +0.208 ± nan | 1 |
| `strip_formula` | +0.165 ± nan | +0.099 ± nan | 1 |
| `mask_ruler,strip_formula` | +0.241 ± nan | +0.215 ± nan | 1 |
| `mask_ruler,crop16` | +0.221 ± nan | +0.186 ± nan | 1 |
| `mask_ruler,crop32` | +0.240 ± nan | +0.208 ± nan | 1 |
| `orthonorm` | +0.096 ± nan | +0.111 ± nan | 1 |
| `mask_ruler,drop_span` | +0.212 ± nan | +0.176 ± nan | 1 |

Ruler-block null on `orig` (all seeds × draws): -0.003 ± 0.109, 95% band [-0.212, +0.197] — that is ONE draw; the reported mc ρ is a mean over draws, whose block null is +0.001 ± 0.009, 95% band [-0.016, +0.017] (200 ruler permutations, seed 0 scores)

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
