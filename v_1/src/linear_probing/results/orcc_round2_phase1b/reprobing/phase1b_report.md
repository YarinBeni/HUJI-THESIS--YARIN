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
| pv0 | last | 4 | 0.079 | 0.130 (-0.051) | 0.117 (-0.038) | FAILS BOTH |
| pv0 | last | 10 | 0.086 | 0.130 (-0.044) | 0.117 (-0.031) | FAILS BOTH |
| pv0 | last | 15 | 0.086 | 0.130 (-0.044) | 0.117 (-0.031) | FAILS BOTH |
| pv0 | last | 22 | 0.086 | 0.130 (-0.044) | 0.117 (-0.031) | FAILS BOTH |
| pv0 | last | 28 | 0.085 | 0.130 (-0.044) | 0.117 (-0.031) | FAILS BOTH |
| pv0 | mean | 0 | 0.139 | 0.130 (+0.009) | 0.117 (+0.022) | BEATS BOTH |
| pv0 | mean | 4 | 0.134 | 0.130 (+0.005) | 0.117 (+0.017) | BEATS BOTH |
| pv0 | mean | 10 | 0.112 | 0.130 (-0.018) | 0.117 (-0.005) | FAILS BOTH |
| pv0 | mean | 15 | 0.091 | 0.130 (-0.038) | 0.117 (-0.026) | FAILS BOTH |
| pv0 | mean | 22 | 0.087 | 0.130 (-0.042) | 0.117 (-0.029) | FAILS BOTH |
| pv0 | mean | 28 | 0.093 | 0.130 (-0.036) | 0.117 (-0.024) | FAILS BOTH |
| pv1 | last | 0 | 0.022 | 0.130 (-0.107) | 0.117 (-0.094) | FAILS BOTH |
| pv1 | last | 4 | 0.075 | 0.130 (-0.054) | 0.117 (-0.041) | FAILS BOTH |
| pv1 | last | 10 | 0.083 | 0.130 (-0.047) | 0.117 (-0.034) | FAILS BOTH |
| pv1 | last | 15 | 0.084 | 0.130 (-0.045) | 0.117 (-0.033) | FAILS BOTH |
| pv1 | last | 22 | 0.080 | 0.130 (-0.050) | 0.117 (-0.037) | FAILS BOTH |
| pv1 | last | 28 | 0.076 | 0.130 (-0.054) | 0.117 (-0.041) | FAILS BOTH |
| pv1 | mean | 0 | 0.139 | 0.130 (+0.009) | 0.117 (+0.022) | BEATS BOTH |
| pv1 | mean | 4 | 0.122 | 0.130 (-0.008) | 0.117 (+0.005) | BEATS MEAN |
| pv1 | mean | 10 | 0.090 | 0.130 (-0.040) | 0.117 (-0.027) | FAILS BOTH |
| pv1 | mean | 15 | 0.086 | 0.130 (-0.044) | 0.117 (-0.031) | FAILS BOTH |
| pv1 | mean | 22 | 0.081 | 0.130 (-0.049) | 0.117 (-0.036) | FAILS BOTH |
| pv1 | mean | 28 | 0.084 | 0.130 (-0.046) | 0.117 (-0.033) | FAILS BOTH |
| pv2 | last | 0 | 0.022 | 0.130 (-0.107) | 0.117 (-0.094) | FAILS BOTH |
| pv2 | last | 4 | 0.077 | 0.130 (-0.053) | 0.117 (-0.040) | FAILS BOTH |
| pv2 | last | 10 | 0.084 | 0.130 (-0.046) | 0.117 (-0.033) | FAILS BOTH |
| pv2 | last | 15 | 0.086 | 0.130 (-0.044) | 0.117 (-0.031) | FAILS BOTH |
| pv2 | last | 22 | 0.085 | 0.130 (-0.045) | 0.117 (-0.032) | FAILS BOTH |
| pv2 | last | 28 | 0.080 | 0.130 (-0.049) | 0.117 (-0.037) | FAILS BOTH |
| pv2 | mean | 0 | 0.139 | 0.130 (+0.009) | 0.117 (+0.022) | BEATS BOTH |
| pv2 | mean | 4 | 0.118 | 0.130 (-0.011) | 0.117 (+0.001) | BEATS MEAN |
| pv2 | mean | 10 | 0.093 | 0.130 (-0.036) | 0.117 (-0.023) | FAILS BOTH |
| pv2 | mean | 15 | 0.092 | 0.130 (-0.038) | 0.117 (-0.025) | FAILS BOTH |
| pv2 | mean | 22 | 0.084 | 0.130 (-0.046) | 0.117 (-0.033) | FAILS BOTH |
| pv2 | mean | 28 | 0.073 | 0.130 (-0.057) | 0.117 (-0.044) | FAILS BOTH |
| pv3 | last | 0 | 0.022 | 0.130 (-0.107) | 0.117 (-0.094) | FAILS BOTH |
| pv3 | last | 4 | 0.073 | 0.130 (-0.057) | 0.117 (-0.044) | FAILS BOTH |
| pv3 | last | 10 | 0.082 | 0.130 (-0.047) | 0.117 (-0.034) | FAILS BOTH |
| pv3 | last | 15 | 0.083 | 0.130 (-0.046) | 0.117 (-0.033) | FAILS BOTH |
| pv3 | last | 22 | 0.082 | 0.130 (-0.047) | 0.117 (-0.034) | FAILS BOTH |
| pv3 | last | 28 | 0.079 | 0.130 (-0.050) | 0.117 (-0.037) | FAILS BOTH |
| pv3 | mean | 0 | 0.139 | 0.130 (+0.009) | 0.117 (+0.022) | BEATS BOTH |
| pv3 | mean | 4 | 0.122 | 0.130 (-0.008) | 0.117 (+0.005) | BEATS MEAN |
| pv3 | mean | 10 | 0.090 | 0.130 (-0.040) | 0.117 (-0.027) | FAILS BOTH |
| pv3 | mean | 15 | 0.086 | 0.130 (-0.044) | 0.117 (-0.031) | FAILS BOTH |
| pv3 | mean | 22 | 0.080 | 0.130 (-0.050) | 0.117 (-0.037) | FAILS BOTH |
| pv3 | mean | 28 | 0.083 | 0.130 (-0.046) | 0.117 (-0.033) | FAILS BOTH |

## Phase 1b PLS year results (full numbers)

| variant | pooling | layer | ruler acc | year-raw MAE | year-raw sp | year-log MAE | year-log sp |
|---|---|---|---|---|---|---|---|
| pv0 | last | 0 | 0.233 | n/a | n/a | n/a | n/a |
| pv0 | last | 4 | 0.329 | 83.62 | 0.375 | 0.2969 | 0.292 |
| pv0 | last | 10 | 0.337 | 83.49 | 0.354 | 0.2821 | 0.227 |
| pv0 | last | 15 | 0.338 | 85.49 | 0.333 | 0.2690 | 0.184 |
| pv0 | last | 22 | 0.339 | 88.05 | 0.268 | 0.2869 | 0.149 |
| pv0 | last | 28 | 0.343 | 85.03 | 0.325 | 0.2664 | 0.220 |
| pv0 | mean | 0 | 0.448 | 82.88 | 0.392 | 0.2543 | 0.361 |
| pv0 | mean | 4 | 0.434 | 84.32 | 0.384 | 0.2555 | 0.311 |
| pv0 | mean | 10 | 0.390 | 79.76 | 0.412 | 0.2941 | 0.298 |
| pv0 | mean | 15 | 0.361 | 79.81 | 0.417 | 0.2741 | 0.327 |
| pv0 | mean | 22 | 0.348 | 80.50 | 0.413 | 0.2997 | 0.283 |
| pv0 | mean | 28 | 0.372 | 82.83 | 0.352 | 0.2626 | 0.275 |
| pv1 | last | 0 | 0.233 | n/a | n/a | n/a | n/a |
| pv1 | last | 4 | 0.322 | 83.92 | 0.343 | 0.2942 | 0.215 |
| pv1 | last | 10 | 0.331 | 82.97 | 0.313 | 0.2735 | 0.164 |
| pv1 | last | 15 | 0.335 | 84.80 | 0.292 | 0.2900 | 0.146 |
| pv1 | last | 22 | 0.327 | 86.18 | 0.291 | 0.2440 | 0.181 |
| pv1 | last | 28 | 0.323 | 81.84 | 0.362 | 0.2698 | 0.265 |
| pv1 | mean | 0 | 0.448 | 82.88 | 0.392 | 0.2543 | 0.361 |
| pv1 | mean | 4 | 0.412 | 78.57 | 0.425 | 0.2555 | 0.331 |
| pv1 | mean | 10 | 0.357 | 80.38 | 0.446 | 0.3106 | 0.301 |
| pv1 | mean | 15 | 0.348 | 83.49 | 0.436 | 0.3051 | 0.350 |
| pv1 | mean | 22 | 0.335 | 82.27 | 0.408 | 0.3013 | 0.318 |
| pv1 | mean | 28 | 0.352 | 80.39 | 0.360 | 0.2622 | 0.329 |
| pv2 | last | 0 | 0.233 | n/a | n/a | n/a | n/a |
| pv2 | last | 4 | 0.323 | 85.11 | 0.348 | 0.2924 | 0.233 |
| pv2 | last | 10 | 0.334 | 89.30 | 0.248 | 0.2940 | 0.166 |
| pv2 | last | 15 | 0.339 | 84.47 | 0.316 | 0.2967 | 0.197 |
| pv2 | last | 22 | 0.335 | 85.47 | 0.302 | 0.2786 | 0.188 |
| pv2 | last | 28 | 0.336 | 85.88 | 0.324 | 0.2130 | 0.252 |
| pv2 | mean | 0 | 0.448 | 82.88 | 0.392 | 0.2543 | 0.361 |
| pv2 | mean | 4 | 0.409 | 79.24 | 0.439 | 0.2557 | 0.388 |
| pv2 | mean | 10 | 0.358 | 77.49 | 0.457 | 0.2683 | 0.345 |
| pv2 | mean | 15 | 0.350 | 79.83 | 0.450 | 0.3072 | 0.375 |
| pv2 | mean | 22 | 0.341 | 78.57 | 0.407 | 0.3151 | 0.307 |
| pv2 | mean | 28 | 0.344 | 80.18 | 0.417 | 0.2688 | 0.299 |
| pv3 | last | 0 | 0.233 | n/a | n/a | n/a | n/a |
| pv3 | last | 4 | 0.317 | 84.70 | 0.329 | 0.2934 | 0.214 |
| pv3 | last | 10 | 0.330 | 83.41 | 0.305 | 0.2740 | 0.138 |
| pv3 | last | 15 | 0.331 | 86.11 | 0.288 | 0.2954 | 0.129 |
| pv3 | last | 22 | 0.330 | 86.35 | 0.286 | 0.2391 | 0.171 |
| pv3 | last | 28 | 0.330 | 80.86 | 0.380 | 0.2729 | 0.250 |
| pv3 | mean | 0 | 0.448 | 82.88 | 0.392 | 0.2543 | 0.361 |
| pv3 | mean | 4 | 0.412 | 78.75 | 0.422 | 0.2561 | 0.326 |
| pv3 | mean | 10 | 0.357 | 80.84 | 0.440 | 0.3082 | 0.302 |
| pv3 | mean | 15 | 0.348 | 83.64 | 0.429 | 0.3052 | 0.347 |
| pv3 | mean | 22 | 0.334 | 82.13 | 0.409 | 0.2986 | 0.325 |
| pv3 | mean | 28 | 0.352 | 80.44 | 0.358 | 0.2704 | 0.275 |

## Per-variant verdict (at best ruler layer)

- **pv0**: best pooling=`mean` layer L0 Macro-F1=0.139 -> BEATS BOTH
- **pv1**: best pooling=`mean` layer L0 Macro-F1=0.139 -> BEATS BOTH
- **pv2**: best pooling=`mean` layer L0 Macro-F1=0.139 -> BEATS BOTH
- **pv3**: best pooling=`mean` layer L0 Macro-F1=0.139 -> BEATS BOTH

## Interpretation

4/4 variants beat Qwen-pretrained `last`, 4/4 beat Qwen-pretrained `mean`, 4/4 beat BOTH Qwen baselines. BUT — the full Round-1 leaderboard shows Qwen-pretrained is the WEAKEST method. Apples-to-apples for the diagnostic question 'does prompted Qwen compete with the actually-good methods?' is below.

**Best Phase 1b Macro-F1 = 0.139** (across all variants/poolings/layers).

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
