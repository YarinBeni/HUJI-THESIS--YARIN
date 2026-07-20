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

## Table 4 — E6 cluster structure (all 10 methods, complete)

Full-corpus KMeans at k=8 aligned (ARI) against each metadata partition; MC =
200 balanced draws (8 rulers x 21). best-k = silhouette-optimal cluster count.
Per-model JSONs: e6_clusters/results/; internal indices: csv/e6_cluster_indices.csv;
k-sweep figures: e6_clusters/figures/.

| model | clean | best-k (mode over 200 draws) | MC ARI(ruler) | ARI ruler | ARI prov | ARI year | k8 aligns best with |
|---|---|---|---|---|---|---|---|
| cunei-400m | maximal | 2 (200/200) | 0.112 | 0.113 | 0.140 | 0.099 | **provenance** |
| cunei-400m | engtier0 | 2 (195/200) | 0.146 | 0.123 | 0.108 | 0.103 | **ruler8** |
| Qwen3-32B | maximal | 2 (199/200) | 0.073 | 0.071 | 0.100 | 0.066 | **provenance** |
| Qwen3-32B | engtier0 | 2 (200/200) | 0.084 | 0.040 | 0.035 | 0.040 | **ruler8** |
| Qwen3-8B | maximal | 2 (118/200) | 0.027 | 0.034 | 0.033 | 0.031 | ruler8 |
| Qwen3-8B | engtier0 | 2 (200/200) | 0.051 | 0.030 | 0.019 | 0.017 | ruler8 |
| Qwen3-1.7B | maximal | 2 (123/200) | 0.027 | 0.027 | 0.030 | 0.025 | provenance |
| Qwen3-1.7B | engtier0 | 2 (200/200) | 0.046 | 0.022 | 0.022 | 0.016 | provenance |
| gpt-oss-120B | maximal | 2 (189/200) | 0.038 | 0.037 | 0.059 | -0.001 | **provenance** |
| gpt-oss-120B | engtier0 | 2 (200/200) | 0.017 | 0.019 | 0.029 | 0.011 | provenance |
| Thal-AKK | maximal | 2 (197/200) | 0.042 | 0.027 | 0.049 | 0.015 | **provenance** |
| Thal-AKK | engtier0 | 2 (200/200) | 0.016 | 0.015 | 0.005 | 0.007 | ruler8 |
| uMT5 | maximal | 2 (188/200) | 0.031 | 0.018 | 0.057 | -0.011 | **provenance** |
| uMT5 | engtier0 | 2 (197/200) | 0.036 | 0.045 | 0.031 | 0.024 | ruler8 |
| MLM | maximal | 2 (200/200) | 0.060 | 0.065 | 0.081 | 0.063 | **provenance** |
| random* | maximal | 12 (36/200) | 0.091 | 0.081 | 0.080 | 0.061 | ruler8 |
| random* | engtier0 | 2 (105/200) | 0.062 | 0.062 | 0.065 | 0.025 | provenance |
| TF-IDF* | maximal | 12 (66/200) | 0.106 | 0.077 | 0.105 | 0.057 | **provenance** |
| TF-IDF* | engtier0 | 3 (69/200) | 0.107 | 0.085 | 0.051 | 0.074 | ruler8 |

**Findings:**
1. **k=8 is essentially never the natural cluster count.** Every trained model's
   silhouette-optimal k is 2 (the Babylonian-vs-Assyrian era split) in ~190-200
   of the 200 balanced draws, where 8 rulers x 21 is literally the ground truth.
   Only the surface controls (TF-IDF, random) ever prefer higher k, and never 8.
2. **No model clusters by ruler above the ~0.11 control level.** cunei-400m (the
   deck's best probe) is the only model at/above TF-IDF's MC ARI; the big Qwen
   LLMs sit at 0.03-0.08 — BELOW the character-n-gram control.
3. **Where clusters align at all, it is PROVENANCE (find-spot), not ruler or
   year** — true for cunei/Qwen3-32B/gpt-oss/Thal-AKK/uMT5/MLM/TF-IDF on Akkadian
   maximal. The embeddings organize by where tablets were excavated.
4. **Internal quality vs metadata alignment are ANTI-correlated** (see
   csv/e6_cluster_indices.csv): Qwen3-8B/1.7B have the highest silhouette (0.30,
   inertia ~2-4) yet the lowest ruler-ARI (0.027) — tight blobs of near-duplicate
   texts, not authorship structure. cunei-400m is the reverse: low silhouette
   (0.05-0.08), best metadata alignment.

## Why the E6 "winners" differ from the deck (Table 1) winners

Same ORCC data, same 200 balanced draws — DIFFERENT question. The deck probes
ask "can a supervised probe READ the year?" (uses the labels); E6 silhouette
asks "does KMeans find tight blobs?" (unsupervised, year never enters). A model
can score high on one and low on the other: Qwen3-8B/1.7B cluster beautifully
(silhouette 0.30) into duplicate-text blobs that carry no chronology (ruler-ARI
0.027), while cunei-400m reads year best (deck rho 0.391) from a diffuse,
low-silhouette space. Best-clusterer != best-year-decoder; internal cluster
quality is a red herring for the world-model question.
