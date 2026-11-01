# Controls wave 2 — length/names, categories-vs-order, target choice

## V1 — language alignment on [mask_ruler + crop32] views

| model | layer | ARI orig | ARI masked+cropped | shuffle null |
|---|---|---|---|---|
| `Llama-2-7b-hf` | 20 | 0.903 | 0.066 | -0.006 |
| `Qwen3-8B` | 27 | 0.532 | 0.124 | -0.003 |
| `cuneiformBase-400m` | 4 | 0.308 | 0.258 | -0.002 |
| `OLMo-2-1124-7B` | 9 | 0.875 | -0.005 | -0.015 |

## V3 — time as categories vs time as order

| model | layer | ARI(clusters, century) | centuries with order test | mean within-century rho |
|---|---|---|---|---|
| `Llama-2-7b-hf` | 20 | -0.064 | 3 | 0.631 |
| `Qwen3-8B` | 27 | 0.016 | 3 | 0.617 |
| `cuneiformBase-400m` | 4 | 0.072 | 3 | 0.701 |
| `OLMo-2-1124-7B` | 9 | -0.062 | 3 | 0.651 |

## V4 — name-probe target: first vs median vs last attested year

Reign is a range; the probe target was the median attested year. Same activations, three target conventions.

| arm | dataset | target=first | median | last |
|---|---|---|---|---|
| llama2_7b | assyrian_ruler_akk | 0.227 | 0.224 | 0.224 |
| llama2_7b | assyrian_ruler | 0.396 | 0.396 | 0.396 |
| qwen3_8b | assyrian_ruler_akk | 0.368 | 0.373 | 0.373 |
| qwen3_8b | assyrian_ruler | 0.535 | 0.536 | 0.536 |
| thalesian_cunei400m | assyrian_ruler_akk | 0.519 | 0.526 | 0.526 |
| thalesian_cunei400m | assyrian_ruler | 0.420 | 0.416 | 0.416 |
