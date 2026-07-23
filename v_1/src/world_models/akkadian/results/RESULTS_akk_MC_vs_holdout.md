# Akkadian G&T-mimic — balanced-MC vs. former holdout (Spearman ρ, site=last)

**holdout**=within-ruler 80/20 (old, ruler-biased). **mc**=balanced Monte-Carlo (cap/ruler, 200 draws, StratifiedKFold-by-ruler; confound removed). **loro**=leave-one-ruler-out (unseen ruler).  r40 mc=nan by construction (single-fragment rulers).

## akk_maximal · year · r8
| arm | holdout | mc | loro |
|---|---|---|---|
| tfidf | +0.793 | +0.707 | +0.129 |
| random | +0.482 | +0.499 | +0.048 |
| qwen3_1b7 | +0.459 | +0.370 | +0.055 |
| qwen3_8b | +0.533 | +0.396 | +0.065 |
| qwen3_32b | +0.523 | +0.312 | +0.091 |
| llama2_7b | +0.591 | +0.433 | -0.067 |
| llama2_7b_random | +0.495 | +0.438 | -0.015 |
| llama2_13b | +0.555 | +0.403 | -0.056 |
| llama2_13b_random | +0.472 | +0.369 | -0.022 |
| llama2_70b | +0.596 | +0.331 | +0.024 |
| llama2_70b_random | +0.409 | +0.305 | -0.117 |

## akk_maximal · year · r40
| arm | holdout | mc | loro |
|---|---|---|---|
| tfidf | +0.539 | nan | +0.171 |
| random | +0.372 | nan | +0.164 |
| qwen3_1b7 | +0.299 | nan | +0.075 |
| qwen3_8b | +0.275 | nan | +0.102 |
| qwen3_32b | +0.273 | nan | +0.151 |
| llama2_7b | +0.312 | nan | +0.044 |
| llama2_7b_random | +0.298 | nan | +0.024 |
| llama2_13b | +0.295 | nan | +0.126 |
| llama2_13b_random | +0.302 | nan | +0.079 |
| llama2_70b | +0.285 | nan | +0.108 |
| llama2_70b_random | +0.248 | nan | +0.012 |

## akk_maximal · geo · r8
| arm | holdout | mc | loro |
|---|---|---|---|
| tfidf | +0.426 | +0.413 | +0.318 |
| random | +0.280 | +0.263 | +0.200 |
| qwen3_1b7 | +0.385 | +0.215 | +0.194 |
| qwen3_8b | +0.425 | +0.262 | +0.206 |
| qwen3_32b | +0.368 | +0.159 | +0.210 |
| llama2_7b | +0.435 | +0.301 | +0.296 |
| llama2_7b_random | +0.284 | +0.285 | +0.188 |
| llama2_13b | +0.411 | +0.255 | +0.231 |
| llama2_13b_random | +0.224 | +0.194 | +0.118 |
| llama2_70b | +0.390 | +0.269 | +0.241 |
| llama2_70b_random | +0.165 | +0.055 | +0.099 |

## akk_maximal · geo · r40
| arm | holdout | mc | loro |
|---|---|---|---|
| tfidf | +0.493 | nan | +0.429 |
| random | +0.454 | nan | +0.284 |
| qwen3_1b7 | +0.434 | nan | +0.294 |
| qwen3_8b | +0.376 | nan | +0.286 |
| qwen3_32b | +0.415 | nan | +0.315 |
| llama2_7b | +0.406 | nan | +0.376 |
| llama2_7b_random | +0.304 | nan | +0.272 |
| llama2_13b | +0.309 | nan | +0.307 |
| llama2_13b_random | +0.323 | nan | +0.180 |
| llama2_70b | +0.417 | nan | +0.330 |
| llama2_70b_random | +0.310 | nan | +0.194 |

## eng_maximal · year · r8
| arm | holdout | mc | loro |
|---|---|---|---|
| tfidf | +0.669 | +0.616 | +0.044 |
| random | +0.530 | +0.425 | +0.120 |
| qwen3_1b7 | +0.464 | +0.369 | +0.059 |
| qwen3_8b | +0.532 | +0.427 | +0.088 |
| qwen3_32b | +0.520 | +0.428 | +0.132 |
| llama2_7b | +0.538 | +0.447 | +0.002 |
| llama2_7b_random | +0.503 | +0.376 | +0.102 |
| llama2_13b | +0.562 | +0.378 | +0.053 |
| llama2_13b_random | +0.447 | +0.389 | +0.028 |
| llama2_70b | +0.577 | +0.428 | +0.064 |
| llama2_70b_random | +0.284 | +0.245 | -0.092 |

## eng_maximal · year · r40
| arm | holdout | mc | loro |
|---|---|---|---|
| tfidf | +0.421 | nan | -0.093 |
| random | +0.310 | nan | +0.107 |
| qwen3_1b7 | +0.311 | nan | +0.075 |
| qwen3_8b | +0.307 | nan | +0.040 |
| qwen3_32b | +0.251 | nan | +0.003 |
| llama2_7b | +0.247 | nan | -0.099 |
| llama2_7b_random | +0.358 | nan | +0.111 |
| llama2_13b | +0.209 | nan | -0.041 |
| llama2_13b_random | +0.453 | nan | +0.088 |
| llama2_70b | +0.188 | nan | -0.010 |
| llama2_70b_random | +0.183 | nan | +0.065 |

## eng_maximal · geo · r8
| arm | holdout | mc | loro |
|---|---|---|---|
| tfidf | +0.428 | +0.393 | +0.256 |
| random | +0.324 | +0.252 | +0.236 |
| qwen3_1b7 | +0.406 | +0.215 | +0.202 |
| qwen3_8b | +0.435 | +0.223 | +0.208 |
| qwen3_32b | +0.363 | +0.201 | +0.217 |
| llama2_7b | +0.422 | +0.275 | +0.272 |
| llama2_7b_random | +0.268 | +0.245 | +0.149 |
| llama2_13b | +0.460 | +0.234 | +0.229 |
| llama2_13b_random | +0.249 | +0.200 | +0.130 |
| llama2_70b | +0.416 | +0.299 | +0.259 |
| llama2_70b_random | +0.058 | +0.046 | +0.026 |

## eng_maximal · geo · r40
| arm | holdout | mc | loro |
|---|---|---|---|
| tfidf | +0.505 | nan | +0.307 |
| random | +0.353 | nan | +0.269 |
| qwen3_1b7 | +0.414 | nan | +0.245 |
| qwen3_8b | +0.433 | nan | +0.241 |
| qwen3_32b | +0.390 | nan | +0.265 |
| llama2_7b | +0.477 | nan | +0.328 |
| llama2_7b_random | +0.290 | nan | +0.187 |
| llama2_13b | +0.461 | nan | +0.270 |
| llama2_13b_random | +0.304 | nan | +0.185 |
| llama2_70b | +0.418 | nan | +0.311 |
| llama2_70b_random | +0.194 | nan | +0.117 |

