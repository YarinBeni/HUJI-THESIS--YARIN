# WA — the Gurnee & Tegmark protocol on Akkadian (rulers & find-spots)

Sibling of the English `world_models` replication: same recipe (build an entity
string → pull its last-token embedding → ridge-probe a year or a coordinate), but the
entities are **Akkadian text fragments** and the targets are their **composition year**
(G&T's headline/figure analog) and their **find-spot (lon, lat)** (their world_place
analog). Reuses `../wm_lib` (registry, model loading, pooling, probing).

## Design

- **Entity = a whole fragment** (G&T's *headline* protocol, not the name protocol),
  in two text variants:
  - `akk_maximal` — the maximal-cleaned Akkadian (`text_maximal`)
  - `eng_maximal` — its English translation (`eng_maximal`) → the translation probe
- **Two ruler sets** (both requested):
  - `r8` — the 8 best-attested rulers (≥20 dated texts, ~1071 frags): dense & clean
  - `r40` — all 40 rulers with a year (~1187 frags): the full, sparse tail
- **Two targets:** `year` (regression) and `geo` (lon, lat from the find-spot via
  `../../stress_tests/shared/sites_gazetteer.csv`; ~1167 frags have coords).
- **Models:** decoder LLMs only — Qwen3 1.7/8/32B, gpt-oss-120B, Llama-2 7/13/70B
  (trained + random), and the `random` (Qwen3-8B from-config) control. **Encoders
  (AKK-300M, cunei-400M, uMT5) are excluded by design** — they have no causal last
  token, so the G&T last-token protocol doesn't apply. Plus a TF-IDF text floor.
- **Pooling:** `last` (paper-faithful) and `mean` (kept for reference); `last` is
  canonical in the tables.
- **Three probe modes** (because ORCC `year` is constant per ruler → year probing ≈
  ruler identification; see thesis `shared/mc_maxking.py`):
  - **holdout** — within-ruler 80/20 split, per layer. G&T-comparable but inflated by
    ruler identity (rulers seen in train & test).
  - **mc** — balanced Monte-Carlo: cap = min ruler count (21 for r8), **200 draws**,
    StratifiedKFold-by-ruler within each draw, at the holdout-best layer. Removes the
    ruler-frequency imbalance; in-distribution. (r40 N/A: min count = 1.)
  - **loro** — **leave-one-ruler-out** (train on 7, predict the held-out ruler; pool
    OOF; swept over layers). The real "place an *unseen* ruler" test — the thesis's
    `year_group`/GroupKFold-by-ruler analog. Spearman is the headline.
  Ridge (alpha = n_features, paper heuristic), R² + Spearman, haversine for geo.

Every (method × variant) is extracted once over all fragments; the probe then slices
to r8/r40 and year/geo. So: **extract = methods × 2 variants; probe = × 2 ruler sets ×
2 targets × 2 sites.**

## Run order (cluster)

```bash
mkdir -p v_1/src/world_models/akkadian/logs
# extraction (parallel)
sbatch v_1/src/world_models/akkadian/sbatch/WA1_extract.sbatch          # gpu:1, 8 methods × 2 variants
sbatch v_1/src/world_models/akkadian/sbatch/WA1b_extract_gptoss.sbatch  # gpu:8
sbatch v_1/src/world_models/akkadian/sbatch/WA1d_extract_llama70b.sbatch# gpu:4 (reuses W0's random-70b checkpoint)
# probing (CPU; per arm once its extraction landed)
sbatch v_1/src/world_models/akkadian/sbatch/WA2_probe.sbatch
sbatch v_1/src/world_models/akkadian/sbatch/WA2b_tfidf.sbatch
# tables
sbatch v_1/src/world_models/akkadian/sbatch/WA3_aggregate.sbatch
```

Local (no GPU): `python tfidf_akk.py`, `python aggregate_akk.py`.

## Outputs

`results/probes/<method>/<variant>.<ruler_set>.<target>.<site>.ridge.json`,
per-target×ruler-set summary CSVs, and `results/RESULTS_akk.md`. `*.npz` activations
are gitignored/cluster-local.

## Reading it (and the honest caveats)

The question: does an Akkadian text's embedding linearly encode *when* and *where* it
was written — and (mirroring the English result) do the small translation encoders /
the trained models beat the surface-form floor?

- **`r8` is the trustworthy panel** (8 dense rulers). **`r40` is harder**: many rulers
  have 1–few texts, so held-out-by-ruler generalization is genuinely hard and scores
  drop — informative, but read it as a tail-generalization test, not a failure.
- **Year with few rulers ≈ ordinal separation**, not continuous regression — a high R²
  can mean "separates the rulers," the categorical-feature confound G&T themselves
  flagged. Hold-one-ruler-out (not run here) is the stronger follow-up.
- **Geo extent is small** (Mesopotamian find-spots), so haversine R² behaves
  differently than a global map; several sites also cluster by province.
- **TF-IDF floor already run** (`python tfidf_akk.py`): akk r8/year R² .634, r40/year
  .171; akk r8/geo .163, r40/geo .317. Embedding arms must clear these.
