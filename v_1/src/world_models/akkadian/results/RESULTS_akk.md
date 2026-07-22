# WA results — Akkadian space & time, G&T protocol

Best-layer held-out **test** scores. Entity = whole fragment (last-token for decoders); target = year or find-spot (lon,lat); held-out-by-ruler split. Encoders excluded (no causal last token).

## year — R²

| row                 |    r8 |   r40 |
|:--------------------|------:|------:|
| tfidf · akk_maximal | 0.634 | 0.171 |
| tfidf · eng_maximal | 0.495 | 0.024 |

## year — Spearman ρ

| row                 |    r8 |   r40 |
|:--------------------|------:|------:|
| tfidf · akk_maximal | 0.793 | 0.539 |
| tfidf · eng_maximal | 0.669 | 0.421 |

## geo — R²

| row                 |    r8 |   r40 |
|:--------------------|------:|------:|
| tfidf · akk_maximal | 0.163 | 0.317 |
| tfidf · eng_maximal | 0.102 | 0.269 |

## geo — Spearman ρ

| row                 |    r8 |   r40 |
|:--------------------|------:|------:|
| tfidf · akk_maximal | 0.426 | 0.493 |
| tfidf · eng_maximal | 0.428 | 0.505 |

