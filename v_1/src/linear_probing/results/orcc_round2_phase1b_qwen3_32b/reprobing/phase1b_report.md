# Phase 1b — Prompted-Activation Re-probing Report

Probes the SAME pipelines (CLS ruler, PLS year raw/log) used in Round 1
on activations extracted with Q+A prompt wrapping (4 variants x 3 layers).
Phase 1b pools at the last token of the fragment span, so the apples-to-
apples Round-1 reference is `last`. We also show `mean` (the headline
Round-1 number, ~0.117 Macro-F1) for context.

## Round-1 baselines — ALL methods, raw fragments, imbalanced

Phase 1b's apples-to-apples comparison is the **same pooling** column.
Best Phase 1b score (~0.139) only beats Qwen-pretrained — it loses badly
to MLM and Random. The headline Round-1 ranking still stands:
**TF-IDF >> Random ≈ MLM > Qwen-pretrained**.

| Method (cleaning, pooling) | best layer | Macro-F1 |
|---|---|---|
| `tfidf__tier0__na__ruler` | 0 | 0.326 |
| `random__tier0__mean__ruler` | 1 | 0.235 |
| `tfidf__maximal__na__ruler` | 0 | 0.228 |
| `mlm__tier0__mean__ruler` | 0 | 0.220 |
| `random__maximal__mean__ruler` | 3 | 0.216 |
| `random__maximal__last__ruler` | 1 | 0.175 |
| `qwen__maximal__last__ruler` | 7 | 0.155 |
| `random__tier0__last__ruler` | 1 | 0.149 |
| `qwen__tier0__last__ruler` | 5 | 0.130 |
| `qwen__maximal__mean__ruler` | 0 | 0.118 |
| `qwen__tier0__mean__ruler` | 0 | 0.117 |

_Best Phase 1b prompted ruler Macro-F1 = 0.139 (mean L0, all 4 variants)._

## Round-1 Qwen Y-baselines (year regression)

- year-raw (last) MAE: 131.13  Spearman: 0.084
- year-raw (mean) MAE: 128.34  Spearman: 0.121
- year-log (last) MAE: 0.4514  Spearman: 0.059
- year-log (mean) MAE: 0.4002  Spearman: 0.095

## Phase 1b ruler results (vs Qwen-pretrained baselines only)

_Note: 'BEATS BOTH' here only means beats Qwen-pretrained. See the_
_Round-1 leaderboard above for the full picture — Phase 1b still loses_
_to MLM (0.220), Random (0.235), and TF-IDF (0.326)._

| variant | pooling | layer | macro_f1 | R1 last (Δ) | R1 mean (Δ) | verdict |
|---|---|---|---|---|---|---|
| pv0 | last | 0 | 0.022 | 0.130 (-0.107) | 0.117 (-0.094) | FAILS BOTH |
| pv0 | last | 16 | 0.080 | 0.130 (-0.049) | 0.117 (-0.036) | FAILS BOTH |
| pv0 | last | 32 | 0.088 | 0.130 (-0.042) | 0.117 (-0.029) | FAILS BOTH |
| pv0 | last | 48 | 0.086 | 0.130 (-0.043) | 0.117 (-0.030) | FAILS BOTH |
| pv0 | last | 63 | 0.061 | 0.130 (-0.069) | 0.117 (-0.056) | FAILS BOTH |
| pv0 | mean | 0 | 0.133 | 0.130 (+0.003) | 0.117 (+0.016) | BEATS BOTH |
| pv0 | mean | 16 | 0.078 | 0.130 (-0.051) | 0.117 (-0.038) | FAILS BOTH |
| pv0 | mean | 32 | 0.094 | 0.130 (-0.035) | 0.117 (-0.022) | FAILS BOTH |
| pv0 | mean | 48 | 0.088 | 0.130 (-0.042) | 0.117 (-0.029) | FAILS BOTH |
| pv0 | mean | 63 | 0.084 | 0.130 (-0.046) | 0.117 (-0.033) | FAILS BOTH |
| pv1 | last | 0 | 0.022 | 0.130 (-0.107) | 0.117 (-0.094) | FAILS BOTH |
| pv1 | last | 16 | 0.067 | 0.130 (-0.063) | 0.117 (-0.050) | FAILS BOTH |
| pv1 | last | 32 | 0.076 | 0.130 (-0.053) | 0.117 (-0.040) | FAILS BOTH |
| pv1 | last | 48 | 0.088 | 0.130 (-0.041) | 0.117 (-0.028) | FAILS BOTH |
| pv1 | last | 63 | 0.065 | 0.130 (-0.064) | 0.117 (-0.052) | FAILS BOTH |
| pv1 | mean | 0 | 0.133 | 0.130 (+0.003) | 0.117 (+0.016) | BEATS BOTH |
| pv1 | mean | 16 | 0.068 | 0.130 (-0.062) | 0.117 (-0.049) | FAILS BOTH |
| pv1 | mean | 32 | 0.081 | 0.130 (-0.048) | 0.117 (-0.036) | FAILS BOTH |
| pv1 | mean | 48 | 0.084 | 0.130 (-0.046) | 0.117 (-0.033) | FAILS BOTH |
| pv1 | mean | 63 | 0.081 | 0.130 (-0.049) | 0.117 (-0.036) | FAILS BOTH |
| pv2 | last | 0 | 0.022 | 0.130 (-0.107) | 0.117 (-0.094) | FAILS BOTH |
| pv2 | last | 16 | 0.075 | 0.130 (-0.055) | 0.117 (-0.042) | FAILS BOTH |
| pv2 | last | 32 | 0.080 | 0.130 (-0.050) | 0.117 (-0.037) | FAILS BOTH |
| pv2 | last | 48 | 0.090 | 0.130 (-0.039) | 0.117 (-0.027) | FAILS BOTH |
| pv2 | last | 63 | 0.062 | 0.130 (-0.068) | 0.117 (-0.055) | FAILS BOTH |
| pv2 | mean | 0 | 0.133 | 0.130 (+0.003) | 0.117 (+0.016) | BEATS BOTH |
| pv2 | mean | 16 | 0.067 | 0.130 (-0.063) | 0.117 (-0.050) | FAILS BOTH |
| pv2 | mean | 32 | 0.083 | 0.130 (-0.046) | 0.117 (-0.034) | FAILS BOTH |
| pv2 | mean | 48 | 0.103 | 0.130 (-0.027) | 0.117 (-0.014) | FAILS BOTH |
| pv2 | mean | 63 | 0.074 | 0.130 (-0.056) | 0.117 (-0.043) | FAILS BOTH |
| pv3 | last | 0 | 0.022 | 0.130 (-0.107) | 0.117 (-0.094) | FAILS BOTH |
| pv3 | last | 16 | 0.065 | 0.130 (-0.064) | 0.117 (-0.052) | FAILS BOTH |
| pv3 | last | 32 | 0.077 | 0.130 (-0.053) | 0.117 (-0.040) | FAILS BOTH |
| pv3 | last | 48 | 0.088 | 0.130 (-0.041) | 0.117 (-0.029) | FAILS BOTH |
| pv3 | last | 63 | 0.068 | 0.130 (-0.062) | 0.117 (-0.049) | FAILS BOTH |
| pv3 | mean | 0 | 0.133 | 0.130 (+0.003) | 0.117 (+0.016) | BEATS BOTH |
| pv3 | mean | 16 | 0.067 | 0.130 (-0.062) | 0.117 (-0.049) | FAILS BOTH |
| pv3 | mean | 32 | 0.082 | 0.130 (-0.047) | 0.117 (-0.035) | FAILS BOTH |
| pv3 | mean | 48 | 0.084 | 0.130 (-0.046) | 0.117 (-0.033) | FAILS BOTH |
| pv3 | mean | 63 | 0.081 | 0.130 (-0.049) | 0.117 (-0.036) | FAILS BOTH |

## Phase 1b PLS year results (full numbers)

| variant | pooling | layer | ruler acc | year-raw MAE | year-raw sp | year-log MAE | year-log sp |
|---|---|---|---|---|---|---|---|
| pv0 | last | 0 | 0.233 | n/a | n/a | n/a | n/a |
| pv0 | last | 16 | 0.337 | 87.23 | 0.332 | 0.2980 | 0.233 |
| pv0 | last | 32 | 0.351 | 82.37 | 0.431 | 0.2967 | 0.300 |
| pv0 | last | 48 | 0.352 | 84.62 | 0.268 | 0.3001 | 0.162 |
| pv0 | last | 63 | 0.298 | 86.16 | 0.347 | 0.2810 | 0.201 |
| pv0 | mean | 0 | 0.440 | 85.73 | 0.359 | 0.2528 | 0.348 |
| pv0 | mean | 16 | 0.350 | 81.27 | 0.440 | 0.2607 | 0.316 |
| pv0 | mean | 32 | 0.365 | 81.49 | 0.398 | 0.2632 | 0.344 |
| pv0 | mean | 48 | 0.361 | 78.15 | 0.453 | 0.3043 | 0.337 |
| pv0 | mean | 63 | 0.360 | 80.99 | 0.388 | 0.2626 | 0.283 |
| pv1 | last | 0 | 0.233 | n/a | n/a | n/a | n/a |
| pv1 | last | 16 | 0.314 | 88.99 | 0.315 | 0.2968 | 0.208 |
| pv1 | last | 32 | 0.326 | 84.85 | 0.373 | 0.2881 | 0.269 |
| pv1 | last | 48 | 0.354 | 86.36 | 0.323 | 0.3098 | 0.227 |
| pv1 | last | 63 | 0.310 | 84.09 | 0.324 | 0.2381 | 0.185 |
| pv1 | mean | 0 | 0.440 | 85.73 | 0.359 | 0.2528 | 0.348 |
| pv1 | mean | 16 | 0.333 | 79.72 | 0.446 | 0.2407 | 0.357 |
| pv1 | mean | 32 | 0.337 | 79.44 | 0.439 | 0.2637 | 0.345 |
| pv1 | mean | 48 | 0.350 | 81.10 | 0.426 | 0.3127 | 0.301 |
| pv1 | mean | 63 | 0.357 | 78.92 | 0.431 | 0.2852 | 0.316 |
| pv2 | last | 0 | 0.233 | n/a | n/a | n/a | n/a |
| pv2 | last | 16 | 0.326 | 92.15 | 0.315 | 0.3122 | 0.259 |
| pv2 | last | 32 | 0.331 | 84.02 | 0.388 | 0.2871 | 0.261 |
| pv2 | last | 48 | 0.357 | 87.39 | 0.320 | 0.3271 | 0.226 |
| pv2 | last | 63 | 0.303 | 78.80 | 0.462 | 0.2371 | 0.225 |
| pv2 | mean | 0 | 0.440 | 85.73 | 0.359 | 0.2528 | 0.348 |
| pv2 | mean | 16 | 0.336 | 80.78 | 0.450 | 0.2529 | 0.377 |
| pv2 | mean | 32 | 0.348 | 79.63 | 0.452 | 0.2680 | 0.360 |
| pv2 | mean | 48 | 0.370 | 82.83 | 0.411 | 0.2963 | 0.262 |
| pv2 | mean | 63 | 0.345 | 80.31 | 0.402 | 0.2684 | 0.292 |
| pv3 | last | 0 | 0.233 | n/a | n/a | n/a | n/a |
| pv3 | last | 16 | 0.311 | 88.70 | 0.310 | 0.2978 | 0.202 |
| pv3 | last | 32 | 0.328 | 84.39 | 0.376 | 0.2914 | 0.267 |
| pv3 | last | 48 | 0.353 | 87.61 | 0.305 | 0.3139 | 0.207 |
| pv3 | last | 63 | 0.316 | 83.29 | 0.343 | 0.2401 | 0.213 |
| pv3 | mean | 0 | 0.440 | 85.73 | 0.359 | 0.2528 | 0.348 |
| pv3 | mean | 16 | 0.332 | 79.70 | 0.447 | 0.2408 | 0.356 |
| pv3 | mean | 32 | 0.338 | 79.33 | 0.440 | 0.2630 | 0.345 |
| pv3 | mean | 48 | 0.349 | 81.06 | 0.427 | 0.3114 | 0.304 |
| pv3 | mean | 63 | 0.356 | 79.12 | 0.431 | 0.2863 | 0.315 |

## Per-variant verdict (at best ruler layer)

- **pv0**: best pooling=`mean` layer L0 Macro-F1=0.133 -> BEATS BOTH
- **pv1**: best pooling=`mean` layer L0 Macro-F1=0.133 -> BEATS BOTH
- **pv2**: best pooling=`mean` layer L0 Macro-F1=0.133 -> BEATS BOTH
- **pv3**: best pooling=`mean` layer L0 Macro-F1=0.133 -> BEATS BOTH

## Interpretation

4/4 variants beat Qwen-pretrained `last`, 4/4 beat Qwen-pretrained `mean`, 4/4 beat BOTH Qwen baselines. BUT — the full Round-1 leaderboard shows Qwen-pretrained is the WEAKEST method. Apples-to-apples for the diagnostic question 'does prompted Qwen compete with the actually-good methods?' is below.

**Best Phase 1b Macro-F1 = 0.133** (across all variants/poolings/layers).

Phase 1b BEATS:
- `qwen__tier0__last__ruler` (0.130)
- `qwen__maximal__mean__ruler` (0.118)
- `qwen__tier0__mean__ruler` (0.117)

Phase 1b LOSES to:
- `tfidf__tier0__na__ruler` (0.326)
- `random__tier0__mean__ruler` (0.235)
- `tfidf__maximal__na__ruler` (0.228)
- `mlm__tier0__mean__ruler` (0.220)
- `random__maximal__mean__ruler` (0.216)
- `random__maximal__last__ruler` (0.175)
- `qwen__maximal__last__ruler` (0.155)
- `random__tier0__last__ruler` (0.149)

Bottom line: prompt-wrapping makes Qwen slightly less bad than its raw form, but the representation is still far behind MLM / Random / TF-IDF. Phase 0 (balanced subsampling) will tell us whether the gap closes under balanced evaluation.
