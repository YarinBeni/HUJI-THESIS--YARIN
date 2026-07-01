# maximal-with-kings scoreboard (balanced-MC, 5 rulers x k=9, king-found)

All three sites on ONE cleaning (clean_maximal_keepking). `ruler_clf` is the control: if `random` matches trained models, the site reads name-token identity, not learned structure. `year_strat` = StratifiedKFold (in-distribution); `year_group` = legacy GroupKFold-by-ruler (degenerate for a per-king-constant label — near 0 by construction).

| model | site | best L | ruler macro-F1 | chance / shuffle | year_strat Sp | year ±10yr acc | year_group Sp |
|---|---|---|---|---|---|---|---|
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