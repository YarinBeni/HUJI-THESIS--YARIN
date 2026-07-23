# Paper (Gurnee & Tegmark) ↔ our experiments — coverage & extension map

## Mechanics
- **Paper:** bare entity string (no prompt), pooling = **last token of the entity span**,
  metric = **R²** on held-out entities. English datasets are balanced → no Monte-Carlo.
- **Us:** two pooling sites in code — `last` (last entity token, paper-faithful, causal
  models) and `mean` (avg over entity tokens; only option for encoders). For Akkadian the
  "entity span" is the **whole fragment**; for English it is the name / headline.

## Table A — dataset analogs
| Paper dataset | Axis | Paper entity → target | Our analog | Our entity → target |
|---|---|---|---|---|
| World Place | space | name → lat/lon | Akkadian find-spot | fragment → provenance lat/lon |
| US Place | space | name → lat/lon | (same geo analog) | — |
| NYC Place | space | POI → lat/lon | (same geo analog) | — |
| Historical Figures | time | name → death year | Akkadian year (name-token) | king-name token → year |
| Media/Art | time | creator's title → year | Akkadian year (title) | fragment → year |
| News Headlines | time | sentence → pub date | **Akkadian year (whole-text)** | **fragment → year** |

3 space datasets → 1 geo analog; 3 time datasets → 1 year analog (Headlines = closest, whole-text→date).

## Table B — pooling × metric coverage
| Experiment | last | mean | Metric | Balancing | Status |
|---|---|---|---|---|---|
| English replication (6 sets) | yes (decoders) | encoders only | R²(+ρ) | held-out (balanced) | have last; ADD mean to decoders |
| Akkadian YEAR (Headlines analog) | yes | yes | R²+ρ | MC 8-ruler cap21 by-ruler | complete |
| Akkadian GEO (Place analog) | yes (by-ruler) | yes (by-ruler) | R² | want by-SITE cap21 | rerun by-site R², last+mean |

## Extensions (agreed)
1. Akkadian: add `last` wherever only `mean` exists (P2/translation geo were mean-only).
2. English: add `mean` to decoders so every dataset shows "last (paper) | mean (ours)".

## Locked jobs
- Geo re-run: 10 sites, cap 21, 200 draws, GroupKFold-by-site → R²(lon/lat), last+mean,
  akk_maximal + eng_tier0, full model set.
- English replication slide: restore before Akkadian slides (optionally add mean column).
- Year: done.
