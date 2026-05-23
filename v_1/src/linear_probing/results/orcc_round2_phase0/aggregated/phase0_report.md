# Phase 0 — Balanced MC Re-probing Gate Report

- Probes dir: `v_1/src/linear_probing/results/orcc_round2_phase0/probes`
- Method tag: `mc_balanced`
- Round-1 CLS baseline: `v_1/src/linear_probing/results/orcc__probe_cls/cls_best_layers.json`
- Round-1 PLS baseline: `v_1/src/linear_probing/results/orcc__probe_pls/pls_best_layers.json`

## Verdict

**Overall: INDETERMINATE**
Missing methods (no MC data): random, thalesian_akk300m, thalesian_cunei400m

Secondary gate (TF-IDF accuracy ≥ 0.70): **FAIL** — value=0.6678 (`tfidf__tier0__na__L00__ruler`)
_run_mc_probes.py uses fit_cls_cv (logistic regression), not k-NN. Treat as proxy only._

## Full Leaderboard — CLS regime (ruler classification)

Sorted by Round-1 (imbalanced) Macro-F1 descending. Balanced MC column shows the best layer found in the Phase-0 layer subset for the same (method, cleaning, pooling). 'n/a' = MC sweep didn't cover that regime.

| Method | Cleaning | Pooling | R1 layer | R1 Macro-F1 | MC layer | MC Macro-F1 (mean ± std) | Δ (MC − R1) |
|---|---|---|---|---|---|---|---|
| TF-IDF | tier0 | na | 0 | 0.3262 | 0 | 0.6496 ± 0.0368 | +0.3234 |
| Thalesian cuneiBase-400m | maximal | mean | 12 | 0.2625 | n/a | n/a | n/a |
| Random-Qwen | tier0 | mean | 1 | 0.2350 | n/a | n/a | n/a |
| TF-IDF | maximal | na | 0 | 0.2277 | 0 | 0.4980 ± 0.0403 | +0.2703 |
| MLM (Aeneas) | tier0 | mean | 0 | 0.2195 | 15 | 0.4604 ± 0.0435 | +0.2408 |
| Random-Qwen | maximal | mean | 3 | 0.2158 | n/a | n/a | n/a |
| Thalesian cuneiBase-400m | tier0 | mean | 12 | 0.2103 | n/a | n/a | n/a |
| Random-Qwen | maximal | last | 1 | 0.1755 | n/a | n/a | n/a |
| Thalesian AKK_300m | tier0 | mean | 8 | 0.1600 | n/a | n/a | n/a |
| Qwen-7B (pretrained) | maximal | last | 7 | 0.1550 | n/a | n/a | n/a |
| Random-Qwen | tier0 | last | 1 | 0.1491 | n/a | n/a | n/a |
| Thalesian AKK_300m | maximal | mean | 8 | 0.1412 | n/a | n/a | n/a |
| Qwen-7B (pretrained) | tier0 | last | 5 | 0.1295 | n/a | n/a | n/a |
| Qwen-7B (pretrained) | maximal | mean | 0 | 0.1182 | n/a | n/a | n/a |
| Qwen-7B (pretrained) | tier0 | mean | 0 | 0.1167 | 0 | 0.3521 ± 0.0417 | +0.2354 |
| Thalesian cuneiBase-400m | maximal | last | 12 | 0.0868 | n/a | n/a | n/a |
| Thalesian cuneiBase-400m | tier0 | last | 12 | 0.0787 | n/a | n/a | n/a |
| Thalesian AKK_300m | tier0 | last | 0 | 0.0222 | n/a | n/a | n/a |
| Thalesian AKK_300m | maximal | last | 0 | 0.0222 | n/a | n/a | n/a |

## Full Leaderboard — PLS regime (ruler via year-PLS-DA)

Sorted by Round-1 (imbalanced) Macro-F1 descending. Balanced MC column shows the best layer found in the Phase-0 layer subset for the same (method, cleaning, pooling). 'n/a' = MC sweep didn't cover that regime.

| Method | Cleaning | Pooling | R1 layer | R1 Macro-F1 | MC layer | MC Macro-F1 (mean ± std) | Δ (MC − R1) |
|---|---|---|---|---|---|---|---|
| Random-Qwen | tier0 | mean | 0 | 0.1147 | n/a | n/a | n/a |
| Thalesian cuneiBase-400m | tier0 | mean | 12 | 0.1143 | n/a | n/a | n/a |
| TF-IDF | tier0 | na | 0 | 0.1128 | 0 | 0.4796 ± 0.0368 | +0.3668 |
| Qwen-7B (pretrained) | tier0 | mean | 0 | 0.1113 | 3 | 0.3632 ± 0.0417 | +0.2520 |
| Thalesian AKK_300m | tier0 | mean | 4 | 0.1081 | n/a | n/a | n/a |
| MLM (Aeneas) | tier0 | mean | 16 | 0.1064 | 14 | 0.3946 ± 0.0423 | +0.2882 |
| TF-IDF | maximal | na | 0 | 0.1013 | 0 | 0.3945 ± 0.0329 | +0.2932 |
| Random-Qwen | tier0 | last | 3 | 0.0989 | n/a | n/a | n/a |
| Qwen-7B (pretrained) | maximal | mean | 1 | 0.0978 | n/a | n/a | n/a |
| Thalesian cuneiBase-400m | maximal | mean | 12 | 0.0943 | n/a | n/a | n/a |
| Random-Qwen | maximal | last | 3 | 0.0941 | n/a | n/a | n/a |
| Qwen-7B (pretrained) | maximal | last | 10 | 0.0906 | n/a | n/a | n/a |
| Random-Qwen | maximal | mean | 19 | 0.0829 | n/a | n/a | n/a |
| Thalesian AKK_300m | tier0 | last | 8 | 0.0821 | n/a | n/a | n/a |
| Thalesian AKK_300m | maximal | mean | 6 | 0.0779 | n/a | n/a | n/a |
| Qwen-7B (pretrained) | tier0 | last | 17 | 0.0779 | n/a | n/a | n/a |
| Thalesian cuneiBase-400m | tier0 | last | 10 | 0.0717 | n/a | n/a | n/a |
| Thalesian AKK_300m | maximal | last | 1 | 0.0562 | n/a | n/a | n/a |
| Thalesian cuneiBase-400m | maximal | last | 10 | 0.0501 | n/a | n/a | n/a |

## TF-IDF (tfidf)
Status: **PASS** (chosen regime: cls, n_draws=200)

| Regime | Cleaning | Pooling | Layer | R1 Macro-F1 | MC mean | MC std | MC median | (mean - 2σ) - R1 | Gate |
|---|---|---|---|---|---|---|---|---|---|
| cls | tier0 | na | 0 | 0.3262 | 0.6496 | 0.0368 | 0.6472 | +0.2497 | PASS |
| pls | tier0 | na | 0 | 0.1128 | 0.4796 | 0.0368 | 0.4802 | +0.2933 | PASS |

## MLM (Aeneas) (mlm)
Status: **PASS** (chosen regime: cls, n_draws=200)

| Regime | Cleaning | Pooling | Layer | R1 Macro-F1 | MC mean | MC std | MC median | (mean - 2σ) - R1 | Gate |
|---|---|---|---|---|---|---|---|---|---|
| cls | tier0 | mean | 0 | 0.2195 | 0.4298 | 0.0371 | 0.4310 | +0.1361 | PASS |
| pls | tier0 | mean | 16 | 0.1064 | 0.3880 | 0.0405 | 0.3891 | +0.2007 | PASS |

## Random-Qwen (random)
Status: **NO DATA** — both CLS and PLS pairings unavailable.
- cls: no MC entry for random__tier0__mean__L01__ruler
- pls: no MC entry for random__tier0__mean__L00__ruler

## Qwen-7B (pretrained) (qwen)
Status: **PASS** (chosen regime: pls, n_draws=200)

| Regime | Cleaning | Pooling | Layer | R1 Macro-F1 | MC mean | MC std | MC median | (mean - 2σ) - R1 | Gate |
|---|---|---|---|---|---|---|---|---|---|
| cls | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | _no MC entry for qwen__maximal__last__L07__ruler_ |
| pls | tier0 | mean | 0 | 0.1113 | 0.3532 | 0.0354 | 0.3515 | +0.1711 | PASS |

## Thalesian AKK_300m (thalesian_akk300m)
Status: **NO DATA** — both CLS and PLS pairings unavailable.
- cls: no MC entry for thalesian_akk300m__tier0__mean__L08__ruler
- pls: no MC entry for thalesian_akk300m__tier0__mean__L04__ruler

## Thalesian cuneiBase-400m (thalesian_cunei400m)
Status: **NO DATA** — both CLS and PLS pairings unavailable.
- cls: no MC entry for thalesian_cunei400m__maximal__mean__L12__ruler
- pls: no MC entry for thalesian_cunei400m__tier0__mean__L12__ruler

## Interpretation

Verdict indeterminate — see missing_methods above.
