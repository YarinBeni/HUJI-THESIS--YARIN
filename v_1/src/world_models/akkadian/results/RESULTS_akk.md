# WA results — Akkadian space & time, G&T protocol

Best-layer held-out **test** scores. Entity = whole fragment (last-token for decoders); target = year or find-spot (lon,lat); held-out-by-ruler split. Encoders excluded (no causal last token).

## year — R²

| row                             |    r8 |   r40 |
|:--------------------------------|------:|------:|
| llama2_70b · eng_maximal        | 0.407 | 0.068 |
| llama2_70b · akk_maximal        | 0.428 | 0.152 |
| llama2_13b · akk_maximal        | 0.417 | 0.114 |
| llama2_13b · eng_maximal        | 0.411 | 0.074 |
| llama2_7b · eng_maximal         | 0.382 | 0.092 |
| llama2_7b · akk_maximal         | 0.418 | 0.122 |
| qwen3_32b · eng_maximal         | 0.336 | 0.081 |
| qwen3_32b · akk_maximal         | 0.316 | 0.146 |
| qwen3_8b · eng_maximal          | 0.297 | 0.057 |
| qwen3_8b · akk_maximal          | 0.386 | 0.091 |
| qwen3_1b7 · eng_maximal         | 0.26  | 0.057 |
| qwen3_1b7 · akk_maximal         | 0.339 | 0.082 |
| llama2_70b_random · eng_maximal | 0.132 | 0.052 |
| llama2_70b_random · akk_maximal | 0.224 | 0.046 |
| llama2_13b_random · akk_maximal | 0.291 | 0.155 |
| llama2_13b_random · eng_maximal | 0.256 | 0.111 |
| llama2_7b_random · akk_maximal  | 0.355 | 0.106 |
| llama2_7b_random · eng_maximal  | 0.351 | 0.083 |
| random · akk_maximal            | 0.343 | 0.145 |
| random · eng_maximal            | 0.338 | 0.085 |
| tfidf · akk_maximal             | 0.634 | 0.171 |
| tfidf · eng_maximal             | 0.495 | 0.024 |

## year — Spearman ρ

| row                             |    r8 |   r40 |
|:--------------------------------|------:|------:|
| llama2_70b · eng_maximal        | 0.577 | 0.188 |
| llama2_70b · akk_maximal        | 0.596 | 0.285 |
| llama2_13b · akk_maximal        | 0.555 | 0.295 |
| llama2_13b · eng_maximal        | 0.562 | 0.209 |
| llama2_7b · eng_maximal         | 0.538 | 0.247 |
| llama2_7b · akk_maximal         | 0.591 | 0.312 |
| qwen3_32b · eng_maximal         | 0.52  | 0.251 |
| qwen3_32b · akk_maximal         | 0.523 | 0.273 |
| qwen3_8b · eng_maximal          | 0.532 | 0.307 |
| qwen3_8b · akk_maximal          | 0.533 | 0.275 |
| qwen3_1b7 · eng_maximal         | 0.464 | 0.311 |
| qwen3_1b7 · akk_maximal         | 0.459 | 0.299 |
| llama2_70b_random · eng_maximal | 0.284 | 0.183 |
| llama2_70b_random · akk_maximal | 0.409 | 0.248 |
| llama2_13b_random · akk_maximal | 0.472 | 0.302 |
| llama2_13b_random · eng_maximal | 0.447 | 0.453 |
| llama2_7b_random · akk_maximal  | 0.495 | 0.298 |
| llama2_7b_random · eng_maximal  | 0.503 | 0.358 |
| random · akk_maximal            | 0.482 | 0.372 |
| random · eng_maximal            | 0.53  | 0.31  |
| tfidf · akk_maximal             | 0.793 | 0.539 |
| tfidf · eng_maximal             | 0.669 | 0.421 |

## geo — R²

| row                             |     r8 |   r40 |
|:--------------------------------|-------:|------:|
| llama2_70b · eng_maximal        |  0.176 | 0.241 |
| llama2_70b · akk_maximal        |  0.148 | 0.165 |
| llama2_13b · akk_maximal        |  0.156 | 0.138 |
| llama2_13b · eng_maximal        |  0.155 | 0.238 |
| llama2_7b · eng_maximal         |  0.165 | 0.245 |
| llama2_7b · akk_maximal         |  0.179 | 0.155 |
| qwen3_32b · eng_maximal         |  0.103 | 0.202 |
| qwen3_32b · akk_maximal         |  0.123 | 0.162 |
| qwen3_8b · eng_maximal          |  0.147 | 0.203 |
| qwen3_8b · akk_maximal          |  0.175 | 0.155 |
| qwen3_1b7 · eng_maximal         |  0.123 | 0.212 |
| qwen3_1b7 · akk_maximal         |  0.139 | 0.155 |
| llama2_70b_random · eng_maximal | -0.031 | 0.096 |
| llama2_70b_random · akk_maximal |  0.018 | 0.164 |
| llama2_13b_random · akk_maximal |  0.087 | 0.192 |
| llama2_13b_random · eng_maximal |  0.049 | 0.177 |
| llama2_7b_random · akk_maximal  |  0.096 | 0.205 |
| llama2_7b_random · eng_maximal  |  0.08  | 0.186 |
| random · akk_maximal            |  0.095 | 0.228 |
| random · eng_maximal            |  0.089 | 0.229 |
| tfidf · akk_maximal             |  0.163 | 0.317 |
| tfidf · eng_maximal             |  0.102 | 0.269 |

## geo — Spearman ρ

| row                             |    r8 |   r40 |
|:--------------------------------|------:|------:|
| llama2_70b · eng_maximal        | 0.416 | 0.418 |
| llama2_70b · akk_maximal        | 0.39  | 0.417 |
| llama2_13b · akk_maximal        | 0.411 | 0.309 |
| llama2_13b · eng_maximal        | 0.46  | 0.461 |
| llama2_7b · eng_maximal         | 0.422 | 0.477 |
| llama2_7b · akk_maximal         | 0.435 | 0.406 |
| qwen3_32b · eng_maximal         | 0.363 | 0.39  |
| qwen3_32b · akk_maximal         | 0.368 | 0.415 |
| qwen3_8b · eng_maximal          | 0.435 | 0.433 |
| qwen3_8b · akk_maximal          | 0.425 | 0.376 |
| qwen3_1b7 · eng_maximal         | 0.406 | 0.414 |
| qwen3_1b7 · akk_maximal         | 0.385 | 0.434 |
| llama2_70b_random · eng_maximal | 0.058 | 0.194 |
| llama2_70b_random · akk_maximal | 0.165 | 0.31  |
| llama2_13b_random · akk_maximal | 0.224 | 0.323 |
| llama2_13b_random · eng_maximal | 0.249 | 0.304 |
| llama2_7b_random · akk_maximal  | 0.284 | 0.304 |
| llama2_7b_random · eng_maximal  | 0.268 | 0.29  |
| random · akk_maximal            | 0.28  | 0.454 |
| random · eng_maximal            | 0.324 | 0.353 |
| tfidf · akk_maximal             | 0.426 | 0.493 |
| tfidf · eng_maximal             | 0.428 | 0.505 |

