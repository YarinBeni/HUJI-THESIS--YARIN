# Stress-test suite — summary tables

Canonical pair everywhere: **Akkadian maximal** (name-stripped) + **English tier0**
(the only valid translation; eng_maximal is broken — the translator hallucinates
king names from name-stripped input). `*` = control. All values from the
committed CSVs in `results/csv/`; protocol = 200 balanced draws, GroupKFold.

## Table 1 — Best models per experiment (top 3 + controls)

| Experiment (metric) | 1st | 2nd | 3rd | TF-IDF* | random* |
|---|---|---|---|---|---|
| Year, Akk maximal — activation PLS (rho) | cunei-400m 0.391 | Qwen3-8B 0.339 | Qwen3-1.7B 0.334 | 0.266 (Ridge) | 0.293 |
| Year, English t0 — activation PLS (rho) | Qwen3-32B 0.437 | Thal-AKK 0.429 | Qwen3-8B 0.416 | 0.349 (Ridge) | 0.275 |
| Geo, Akk maximal — PLS (km, lower better) | cunei-400m 205 | gpt-oss 218 | Qwen3-8B 221 | 254 | 229 |
| Geo, English t0 — PLS (km) | cunei-400m 186 | gpt-oss 196 | Qwen3-32B 201 | 257 | 221 |
| T12 answers, Akk maximal — best prompt (rho) | Qwen3-8B 0.374 (few-shot) | Qwen3-32B 0.304 (few-shot) | gpt-oss 0.299 (expert) | 0.266 floor | 0.292 floor |
| T12 answers, English t0 — best prompt (rho) | Qwen3-32B 0.615 (expert) | gpt-oss 0.595 (CoT) | Qwen3-8B 0.572 (CoT) | 0.349 floor | 0.292 floor |
| P8 dial, Akk maximal — best abs rho(z1,y) | cunei-400m 0.344 (l=.9) | **random 0.305 (l=1)** | **TF-IDF 0.303 (l=.9)** | <- | <- |
| P8 dial, English t0 — best abs rho | Qwen3-32B 0.392 (l=.7) | **TF-IDF 0.389 (l=.9)** | cunei-400m 0.376 (l=1) | <- | random 0.370 |
| P9 RBF-KPLS, Akk maximal (rho) | cunei-400m 0.393 | **TF-IDF 0.313** | Qwen3-32B 0.312 | <- | 0.286 |
| P9 RBF-KPLS, English t0 (rho) | Qwen3-32B 0.406 | cunei-400m 0.379 | uMT5 0.378 | 0.354 | 0.243 |

**Reading**: the only model that ever clears both controls with margin is
cunei-400m (the cuneiform-domain translator encoder). In the geometry probes
(P8/P9) the controls place 2nd-3rd. The big LLMs' only decisive wins are the
T12 English answers — which the ruler-F1 (0.48-0.56) shows are name-carried —
plus Qwen3-8B's few-shot era-anchoring on Akkadian (0.374, the single cell in
the suite that clears both control floors on name-stripped text).

## Table 2 — 2-D map separability scan (763 maximal/engtier0 maps in the GUI data)

Metric: kNN-year readability = |Spearman(year, 10-NN mean-year prediction)| on
the 2-D map; ruler silhouette over the top-8 rulers. Full ranking:
`map_scores.csv` (scan artifacts, not committed).

| Cleaning | Top-3 by kNN-year | Best ruler-silhouette |
|---|---|---|
| Akk maximal | TF-IDF PLS-proj 0.665 *(supervised artifact)* / random-UMAP L23 0.561 *(duplicate islands)* / random-UMAP L14 0.549 | TF-IDF PLSDA -0.02 *(supervised)*; best honest: Qwen2.5 L28 t-SNE -0.06 |
| English t0 | cunei-400m L12 t-SNE 0.560 / Qwen3-32B L60 t-SNE 0.508 / uMT5 L0 t-SNE 0.497 | random L36 PCA -0.09 / Qwen3-8B L32 -0.10 |

All silhouettes are negative: no configuration forms compact ruler/year
clusters in 2-D.

### Visual-inspection verdicts (3-agent pass over 23 maps + Neo-Assyrian follow-up)

* Every visible "year" separation in every map is ONE split: Neo-Babylonian
  (Nebuchadnezzar/Nabonidus, ~539-605) vs Neo-Assyrian (~612-727). No map
  resolves years inside the Assyrian bulk on the full corpus.
* The exotic high scorers are mirages: random-init UMAP scores come from
  ruler-pure duplicate-text islands; TF-IDF's supervised PLS projection only
  peels off extreme-age tails.
* Crispest honest maps: uMT5-base maximal L0 and MLM maximal L0 (one clean
  Neo-Babylonian island each); most gradient-like: Thal-AKK-300m engtier0 L0
  (three era bands in chronological order) and Qwen3-1.7B engtier0 L0.
* **Neo-Assyrian-only re-scan (921 frags, 6 reigns, 727-612)**: best
  unsupervised map = cunei-400m engtier0 L12 t-SNE (4 of 6 reigns form
  identifiable territories) — but the reign regions are t-SNE islands placed
  arbitrarily, NOT in chronological order. Only the supervised TF-IDF PLS-DA
  reference shows a year gradient, and even it cannot separate Esarhaddon from
  Sin-sarru-iskun. Conclusion: embeddings separate WHO (reign identity via
  style/formulae), never WHEN.


## Table 3 — PLS k=3 3-D separability (supervised projection; MC = mean 5-NN-year over the 200 balanced draws)

| Rank | Config | MC-kNN | full-kNN | ruler sil |
|---|---|---|---|---|
| 1 | **TF-IDF* maximal** | **0.707 +- .042** | 0.682 | -0.08 |
| 2 | **TF-IDF* engtier0** | **0.694 +- .041** | 0.718 | -0.07 |
| 3 | Thal-AKK eng-t0 L05 | 0.677 | 0.504 | -0.09 |
| 4 | Qwen3-32B eng-t0 L60 | 0.671 | 0.512 | -0.07 |
| 5 | **random* eng-t0 L36 (untrained)** | **0.651** | 0.648 | -0.07 |
| 6 | cunei-400m eng-t0 L12 | 0.649 | 0.598 | -0.08 |
| 7 | cunei-400m maximal L10 | 0.637 | 0.561 | -0.08 |
| ... | best other Akk-max (gpt-oss L04) | 0.505 | 0.465 | -0.10 |

Verdict: under the IDENTICAL supervised k=3 projection, TF-IDF character
n-grams beat every trained model on both cleanings, and the untrained
random-init ranks 5th overall — the models' activation spaces contain no
year structure beyond what surface statistics provide, in 3-D exactly as in
2-D, full-dim linear, kernel, and spectral probes. All ruler silhouettes
remain negative.

## Table 4 — E6 cluster structure (streaming in from J23)

TF-IDF* baseline: MC ARI(ruler) = 0.106/0.107, k=8 optimal in only 17/200
(maximal) and 7/200 (engtier0) draws, all label silhouettes ~0.
cunei-400m: MC ARI 0.112 (max) / 0.146 (eng); best-k = 2 in 200/200 and
195/200 draws; k8-clusters align better with PROVENANCE (0.140) than rulers
(0.113) on maximal. Full table when J23 completes: results/csv + per-model
JSONs in e6_clusters/results/.
