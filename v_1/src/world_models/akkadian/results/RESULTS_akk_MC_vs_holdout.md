# Akkadian G&T-mimic — balanced-MC vs. the former holdout (Spearman ρ, site=last)

Three read-outs per cell: **holdout** = within-ruler 80/20 (the biased number we had before);
**mc** = balanced Monte-Carlo (cap/ruler, 200 draws, StratifiedKFold-by-ruler — imbalance & the
year==ruler-id confound removed); **loro** = leave-one-ruler-out (place an *unseen* ruler).

> r40 `mc` is **nan by construction**: with all 40 rulers many are single-fragment, so a
> stratified-by-ruler resample is undefined. For r40 read holdout vs **loro** instead.

## akk_maximal · year · r8
| arm | holdout | mc | loro | fmt |
|---|---|---|---|---|
| tfidf | +0.793 | +0.707 | +0.129 | 3mode |
| random | +0.482 | — | — | OLD (holdout only) |
| qwen3_1b7 | +0.459 | — | — | OLD (holdout only) |
| qwen3_8b | +0.533 | — | — | OLD (holdout only) |
| qwen3_32b | +0.523 | — | — | OLD (holdout only) |
| llama2_7b | +0.591 | — | — | OLD (holdout only) |
| llama2_7b_random | +0.495 | — | — | OLD (holdout only) |
| llama2_13b | +0.555 | — | — | OLD (holdout only) |
| llama2_13b_random | +0.472 | — | — | OLD (holdout only) |
| llama2_70b | +0.596 | — | — | OLD (holdout only) |
| llama2_70b_random | +0.409 | +0.305 | -0.117 | 3mode |

## akk_maximal · year · r40
| arm | holdout | mc | loro | fmt |
|---|---|---|---|---|
| tfidf | +0.539 | +nan | +0.171 | 3mode |
| random | +0.372 | — | — | OLD (holdout only) |
| qwen3_1b7 | +0.299 | — | — | OLD (holdout only) |
| qwen3_8b | +0.275 | — | — | OLD (holdout only) |
| qwen3_32b | +0.273 | — | — | OLD (holdout only) |
| llama2_7b | +0.312 | — | — | OLD (holdout only) |
| llama2_7b_random | +0.298 | — | — | OLD (holdout only) |
| llama2_13b | +0.295 | +nan | +0.126 | 3mode |
| llama2_13b_random | +0.302 | +nan | +0.079 | 3mode |
| llama2_70b | +0.285 | +nan | +0.108 | 3mode |
| llama2_70b_random | +0.248 | +nan | +0.012 | 3mode |

## akk_maximal · geo · r8
| arm | holdout | mc | loro | fmt |
|---|---|---|---|---|
| tfidf | +0.426 | +0.413 | +0.318 | 3mode |
| random | +0.280 | — | — | OLD (holdout only) |
| qwen3_1b7 | +0.385 | — | — | OLD (holdout only) |
| qwen3_8b | +0.425 | — | — | OLD (holdout only) |
| qwen3_32b | +0.368 | — | — | OLD (holdout only) |
| llama2_7b | +0.435 | — | — | OLD (holdout only) |
| llama2_7b_random | +0.284 | — | — | OLD (holdout only) |
| llama2_13b | +0.411 | — | — | OLD (holdout only) |
| llama2_13b_random | +0.224 | — | — | OLD (holdout only) |
| llama2_70b | +0.390 | — | — | OLD (holdout only) |
| llama2_70b_random | +0.165 | +0.055 | +0.099 | 3mode |

## akk_maximal · geo · r40
| arm | holdout | mc | loro | fmt |
|---|---|---|---|---|
| tfidf | +0.493 | +nan | +0.429 | 3mode |
| random | +0.454 | +nan | +0.284 | 3mode |
| qwen3_1b7 | +0.434 | — | — | OLD (holdout only) |
| qwen3_8b | +0.376 | +nan | +0.286 | 3mode |
| qwen3_32b | +0.415 | +nan | +0.315 | 3mode |
| llama2_7b | +0.406 | +nan | +0.376 | 3mode |
| llama2_7b_random | +0.304 | +nan | +0.272 | 3mode |
| llama2_13b | +0.309 | +nan | +0.307 | 3mode |
| llama2_13b_random | +0.323 | +nan | +0.180 | 3mode |
| llama2_70b | +0.417 | +nan | +0.330 | 3mode |
| llama2_70b_random | +0.310 | +nan | +0.194 | 3mode |

## eng_maximal · year · r8
| arm | holdout | mc | loro | fmt |
|---|---|---|---|---|
| tfidf | +0.669 | +0.616 | +0.044 | 3mode |
| random | +0.530 | +0.425 | +0.120 | 3mode |
| qwen3_1b7 | +0.464 | — | — | OLD (holdout only) |
| qwen3_8b | +0.532 | +0.427 | +0.088 | 3mode |
| qwen3_32b | +0.520 | +0.428 | +0.132 | 3mode |
| llama2_7b | +0.538 | +0.447 | +0.002 | 3mode |
| llama2_7b_random | +0.503 | +0.376 | +0.102 | 3mode |
| llama2_13b | +0.562 | +0.378 | +0.053 | 3mode |
| llama2_13b_random | +0.447 | +0.389 | +0.028 | 3mode |
| llama2_70b | +0.577 | +0.428 | +0.064 | 3mode |
| llama2_70b_random | +0.284 | +0.245 | -0.092 | 3mode |

## eng_maximal · year · r40
| arm | holdout | mc | loro | fmt |
|---|---|---|---|---|
| tfidf | +0.421 | +nan | -0.093 | 3mode |
| random | +0.310 | +nan | +0.107 | 3mode |
| qwen3_1b7 | +0.311 | +nan | +0.075 | 3mode |
| qwen3_8b | +0.307 | +nan | +0.040 | 3mode |
| qwen3_32b | +0.251 | +nan | +0.003 | 3mode |
| llama2_7b | +0.247 | +nan | -0.099 | 3mode |
| llama2_7b_random | +0.358 | +nan | +0.111 | 3mode |
| llama2_13b | +0.209 | +nan | -0.041 | 3mode |
| llama2_13b_random | +0.453 | +nan | +0.088 | 3mode |
| llama2_70b | +0.188 | +nan | -0.010 | 3mode |
| llama2_70b_random | +0.183 | +nan | +0.065 | 3mode |

## eng_maximal · geo · r8
| arm | holdout | mc | loro | fmt |
|---|---|---|---|---|
| tfidf | +0.428 | +0.393 | +0.256 | 3mode |
| random | +0.324 | +0.252 | +0.236 | 3mode |
| qwen3_1b7 | +0.406 | +0.215 | +0.202 | 3mode |
| qwen3_8b | +0.435 | +0.223 | +0.208 | 3mode |
| qwen3_32b | +0.363 | +0.201 | +0.217 | 3mode |
| llama2_7b | +0.422 | +0.275 | +0.272 | 3mode |
| llama2_7b_random | +0.268 | +0.245 | +0.149 | 3mode |
| llama2_13b | +0.460 | +0.234 | +0.229 | 3mode |
| llama2_13b_random | +0.249 | +0.200 | +0.130 | 3mode |
| llama2_70b | +0.416 | +0.299 | +0.259 | 3mode |
| llama2_70b_random | +0.058 | +0.046 | +0.026 | 3mode |

## eng_maximal · geo · r40
| arm | holdout | mc | loro | fmt |
|---|---|---|---|---|
| tfidf | +0.505 | +nan | +0.307 | 3mode |
| random | +0.353 | +nan | +0.269 | 3mode |
| qwen3_1b7 | +0.414 | +nan | +0.245 | 3mode |
| qwen3_8b | +0.433 | +nan | +0.241 | 3mode |
| qwen3_32b | +0.390 | +nan | +0.265 | 3mode |
| llama2_7b | +0.477 | +nan | +0.328 | 3mode |
| llama2_7b_random | +0.290 | +nan | +0.187 | 3mode |
| llama2_13b | +0.461 | +nan | +0.270 | 3mode |
| llama2_13b_random | +0.304 | +nan | +0.185 | 3mode |
| llama2_70b | +0.418 | +nan | +0.311 | 3mode |
| llama2_70b_random | +0.194 | +nan | +0.117 | 3mode |

