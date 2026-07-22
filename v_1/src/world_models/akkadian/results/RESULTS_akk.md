# WA results — Akkadian, three modes

Per method×text-variant. **r8 holdout R²** (within-ruler split, inflated by ruler identity) → **r8 MC ρ** (balanced, in-distribution, 200 draws) → **r8 LORO ρ** (leave-one-ruler-out — the real 'place an unseen ruler' test). r40 MC is N/A (min ruler count = 1). Decoders last-token; encoders excluded.

## year

| row                             |   r8 hold R² |   r8 MC ρ |   r8 LORO ρ |   r40 hold R² |   r40 LORO ρ |
|:--------------------------------|-------------:|----------:|------------:|--------------:|-------------:|
| llama2_70b · akk_maximal        |        0.428 |   nan     |     nan     |         0.152 |      nan     |
| llama2_70b · eng_maximal        |        0.407 |   nan     |     nan     |         0.068 |      nan     |
| llama2_13b · akk_maximal        |        0.417 |   nan     |     nan     |         0.114 |      nan     |
| llama2_13b · eng_maximal        |        0.411 |   nan     |     nan     |         0.074 |      nan     |
| llama2_7b · akk_maximal         |        0.418 |   nan     |     nan     |         0.122 |      nan     |
| llama2_7b · eng_maximal         |        0.382 |   nan     |     nan     |         0.092 |      nan     |
| qwen3_32b · akk_maximal         |        0.316 |   nan     |     nan     |         0.146 |      nan     |
| qwen3_32b · eng_maximal         |        0.336 |   nan     |     nan     |         0.081 |      nan     |
| qwen3_8b · akk_maximal          |        0.386 |   nan     |     nan     |         0.091 |      nan     |
| qwen3_8b · eng_maximal          |        0.297 |   nan     |     nan     |         0.057 |      nan     |
| qwen3_1b7 · akk_maximal         |        0.339 |   nan     |     nan     |         0.082 |      nan     |
| qwen3_1b7 · eng_maximal         |        0.26  |   nan     |     nan     |         0.057 |      nan     |
| llama2_70b_random · akk_maximal |        0.224 |   nan     |     nan     |         0.046 |      nan     |
| llama2_70b_random · eng_maximal |        0.132 |   nan     |     nan     |         0.052 |      nan     |
| llama2_13b_random · akk_maximal |        0.291 |   nan     |     nan     |         0.155 |      nan     |
| llama2_13b_random · eng_maximal |        0.256 |   nan     |     nan     |         0.111 |      nan     |
| llama2_7b_random · akk_maximal  |        0.355 |   nan     |     nan     |         0.106 |      nan     |
| llama2_7b_random · eng_maximal  |        0.351 |   nan     |     nan     |         0.083 |      nan     |
| random · akk_maximal            |        0.343 |   nan     |     nan     |         0.145 |      nan     |
| random · eng_maximal            |        0.338 |   nan     |     nan     |         0.085 |      nan     |
| tfidf · akk_maximal             |        0.634 |     0.707 |       0.129 |         0.171 |        0.171 |
| tfidf · eng_maximal             |        0.495 |     0.616 |       0.044 |         0.024 |       -0.093 |

## geo

| row                             |   r8 hold R² |   r8 MC ρ |   r8 LORO ρ |   r40 hold R² |   r40 LORO ρ |
|:--------------------------------|-------------:|----------:|------------:|--------------:|-------------:|
| llama2_70b · akk_maximal        |        0.148 |   nan     |     nan     |         0.165 |      nan     |
| llama2_70b · eng_maximal        |        0.176 |   nan     |     nan     |         0.241 |      nan     |
| llama2_13b · akk_maximal        |        0.156 |   nan     |     nan     |         0.138 |      nan     |
| llama2_13b · eng_maximal        |        0.155 |   nan     |     nan     |         0.238 |      nan     |
| llama2_7b · akk_maximal         |        0.179 |   nan     |     nan     |         0.155 |      nan     |
| llama2_7b · eng_maximal         |        0.165 |   nan     |     nan     |         0.245 |      nan     |
| qwen3_32b · akk_maximal         |        0.123 |   nan     |     nan     |         0.162 |      nan     |
| qwen3_32b · eng_maximal         |        0.103 |   nan     |     nan     |         0.202 |      nan     |
| qwen3_8b · akk_maximal          |        0.175 |   nan     |     nan     |         0.155 |      nan     |
| qwen3_8b · eng_maximal          |        0.147 |   nan     |     nan     |         0.203 |      nan     |
| qwen3_1b7 · akk_maximal         |        0.139 |   nan     |     nan     |         0.155 |      nan     |
| qwen3_1b7 · eng_maximal         |        0.123 |   nan     |     nan     |         0.212 |      nan     |
| llama2_70b_random · akk_maximal |        0.018 |   nan     |     nan     |         0.164 |      nan     |
| llama2_70b_random · eng_maximal |       -0.031 |   nan     |     nan     |         0.096 |      nan     |
| llama2_13b_random · akk_maximal |        0.087 |   nan     |     nan     |         0.192 |      nan     |
| llama2_13b_random · eng_maximal |        0.049 |   nan     |     nan     |         0.177 |      nan     |
| llama2_7b_random · akk_maximal  |        0.096 |   nan     |     nan     |         0.205 |      nan     |
| llama2_7b_random · eng_maximal  |        0.08  |   nan     |     nan     |         0.186 |      nan     |
| random · akk_maximal            |        0.095 |   nan     |     nan     |         0.228 |      nan     |
| random · eng_maximal            |        0.089 |   nan     |     nan     |         0.229 |      nan     |
| tfidf · akk_maximal             |        0.163 |     0.413 |       0.318 |         0.317 |        0.429 |
| tfidf · eng_maximal             |        0.102 |     0.393 |       0.256 |         0.269 |        0.307 |

