# maximal-with-kings scoreboard (balanced-MC, 5 rulers x k=9, king-found)

All three sites on ONE cleaning (clean_maximal_keepking). `ruler_clf` is the control: if `random` matches trained models, the site reads name-token identity, not learned structure. `year_strat` = StratifiedKFold (in-distribution); `year_group` = legacy GroupKFold-by-ruler (degenerate for a per-king-constant label — near 0 by construction).

| model | site | best L | ruler macro-F1 | chance / shuffle | year_strat Sp | year ±10yr acc | year_group Sp |
|---|---|---|---|---|---|---|---|
| qwen3_1b7 | mean | L1 | 0.663 | 0.20 / 0.08 | 0.720 | 0.32 | -0.092 |
| qwen3_1b7 | king_last | L3 | 0.979 | 0.20 / 0.08 | 0.977 | 0.98 | 0.056 |
| qwen3_1b7 | king_mean | L0 | 0.965 | 0.20 / 0.07 | 0.972 | 0.94 | -0.562 |
| | | | | | | | |
| qwen3_8b | mean | L0 | 0.706 | 0.20 / 0.08 | 0.715 | 0.32 | -0.060 |
| qwen3_8b | king_last | L3 | 0.989 | 0.20 / 0.08 | 0.974 | 0.97 | -0.055 |
| qwen3_8b | king_mean | L0 | 0.973 | 0.20 / 0.08 | 0.976 | 0.94 | -0.257 |
| | | | | | | | |
| qwen3_32b | mean | L6 | 0.717 | 0.20 / 0.08 | 0.760 | 0.37 | -0.101 |
| qwen3_32b | king_last | L4 | 0.982 | 0.20 / 0.08 | 0.977 | 0.98 | -0.190 |
| qwen3_32b | king_mean | L0 | 0.970 | 0.20 / 0.07 | 0.976 | 0.95 | -0.120 |
| | | | | | | | |
| thalesian_akk300m | mean | L0 | 0.700 | 0.20 / 0.08 | 0.706 | 0.32 | -0.265 |
| thalesian_akk300m | king_last | L7 | 0.975 | 0.20 / 0.08 | 0.962 | 0.90 | -0.228 |
| thalesian_akk300m | king_mean | L2 | 0.998 | 0.20 / 0.08 | 0.975 | 0.96 | -0.582 |
| | | | | | | | |
| thalesian_cunei400m | mean | L1 | 0.897 | 0.20 / 0.06 | 0.851 | 0.48 | 0.029 |
| thalesian_cunei400m | king_last | L12 | 0.943 | 0.20 / 0.08 | 0.957 | 0.92 | -0.148 |
| thalesian_cunei400m | king_mean | L9 | 0.986 | 0.20 / 0.08 | 0.961 | 0.91 | -0.307 |
| | | | | | | | |
| umt5_base | mean | L0 | 0.698 | 0.20 / 0.07 | 0.700 | 0.31 | -0.298 |
| umt5_base | king_last | L2 | 0.953 | 0.20 / 0.07 | 0.968 | 0.96 | -0.343 |
| umt5_base | king_mean | L3 | 0.974 | 0.20 / 0.08 | 0.960 | 0.89 | -0.658 |
| | | | | | | | |
| random | mean | L0 | 0.741 | 0.20 / 0.08 | 0.740 | 0.32 | -0.171 |
| random | king_last | L1 | 0.946 | 0.20 / 0.08 | 0.926 | 0.73 | 0.195 |
| random | king_mean | L0 | 0.971 | 0.20 / 0.08 | 0.981 | 0.96 | -0.547 |
| | | | | | | | |