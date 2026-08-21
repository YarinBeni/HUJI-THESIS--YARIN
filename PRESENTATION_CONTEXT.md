# Presentation context — one file to orient a fresh agent

> **⚠️ PARTIALLY SUPERSEDED — read [`HANDOFF_PROMPT.md`](HANDOFF_PROMPT.md) first.**
> This file was written when the deck had **33** slides; it now has **43** (phase 2 was
> added as slides 32–43, and slides were removed/reordered since). **Section 4's
> slide-by-slide map and the slide counts in section 0 are stale** — use the map in
> `HANDOFF_PROMPT.md` instead. Sections 1–3 (what we extend, the climbing matrix, and
> where the phase-1 code and data live) are still accurate and useful.

**Purpose of this file.** You are being asked to help reorganise the story, order and
slides of a thesis presentation. This file tells you: where the deck is, what every slide
is, which experiment produced it, where that experiment's code and results live, and how
the whole thing extends the paper we are replicating. Read this first; you should not need
to spelunk the repo to get oriented.

Repo root: `HUJI-THESIS--YARIN`. Everything below is relative to it. Branch: `main`.

---

## 0. The deck

| What | Where |
|---|---|
| **Presentation HTML (the deliverable)** | `v_1/src/stress_tests/results/thesis_story_9.html` |
| Slide count | **33** (`data-index` 0–32; on-screen counter is +1) |
| Format | one `<section class="slide ...">` per slide, self-contained, images inlined as base64 |
| Navigation state | `const TOTAL = 33;` and `const TITLES = [...]` near the bottom of the file — **both must be updated when adding/removing/reordering slides** |
| Slide kinds | `slide-title`, `slide-text` (tables), `slide-figure` (base64 image), `slide-method` |
| Common blocks | `.eyebrow` (kicker), `h2.sh` (title), `.cfg` (Setup/Metric config box), `.rtbl.compact` (results table; `tr.rand` = control row), `.fig-wrap` (image), `.takeaway` |

**Editing rule that has bitten before:** sections, `TOTAL`, and `TITLES` must stay in sync,
and `data-index` values must be contiguous from 0. Verify after every edit with:

```bash
python3 - <<'PY'
import re; h=open("v_1/src/stress_tests/results/thesis_story_9.html").read()
print("sections:",h.count("<section"),h.count("</section>"))
print("indices:",re.findall(r'data-index="(\d+)"',h))
print("TOTAL:",re.search(r'const TOTAL = (\d+)',h).group(1))
print("titles:",len(re.findall(r'"(?:[^"\\]|\\.)*"',re.search(r'const TITLES = \[(.*?)\];',h,re.S).group(1))))
PY
```

Other companion docs (shorter, narrower): `v_1/src/world_models/EXPERIMENT_MAPPING.md`
(paper-dataset ↔ our-analog table) and `v_1/src/world_models/EXPERIMENT_MAP_MATRIX.md`
(the matrix + slide map; this file supersedes and expands it).

---

## 1. What we are extending, in one paragraph

Gurnee & Tegmark, *Language Models Represent Space and Time* (2023): probe a frozen LLM's
hidden states with linear ridge regression and recover real-world **space** (lat/lon) and
**time** (year) for named entities. Their setup is one narrow cell: **salient entities**
(famous places, figures, artworks, headlines), a **high-resource language** (English),
**last-token** pooling, **R²** on held-out entities.

Our thesis is about **dating low-resource ancient Akkadian**. So the natural question is
whether their "LLMs build a linear world model" claim survives when you move to *our*
regime — obscure entities, a low-resource language. It does not, and the interesting work
is showing *which* of those two changes is responsible.

---

## 2. The matrix — the organising idea

|                      | **High-resource (English)** | **Low-resource (Akkadian)** |
|----------------------|-----------------------------|-----------------------------|
| **Salient entities** | **CELL A** — the paper's cell | **CELL D — empty** |
| **Obscure entities** | **CELL B** — Assyrian rulers/find-spots written *in English* (`eng_tier0` gloss) | **CELL C** — the same entities in raw Akkadian (`akk_maximal`) |

* **A → C** changes entity salience *and* language simultaneously, so it cannot attribute
  the collapse to either.
* **CELL B is the control that decomposes it**: same entities as C, but in English.
  * **A vs B** isolates **entity obscurity** (language held fixed)
  * **B vs C** isolates **language resource** (entities held fixed)
* **CELL D is empty** and has no natural filler — no famous entities exist in Akkadian
  outside these same royal names. The honest in-data substitute is the attestation
  gradient `r8` (8 best-attested rulers) vs `r40` (the long tail).

### Three categories that must not be conflated

1. **Cells** — genuinely different science (A, B, C).
2. **Variants inside a cell** — pooling (`last` / `mean`), entity span (whole fragment vs
   king-name token), probe family (ridge / PLS-k / kernel PLS / geodesic KPLS /
   supervision dial), model ladder, read-out (R² / Spearman / great-circle km).
3. **Fairness devices** — protocol only, *not findings*: `maximal` cleaning, balanced
   Monte-Carlo (r8, cap 21, 200 draws), by-site MC (10 merged find-spots, cap 21),
   LORO / leave-one-ruler-out, `r8` vs `r40`. These exist because our corpus is unbalanced
   and the paper's is not.

### Coverage

| Cell | time | space | layer sweep | PLS-k | last | mean | random-init controls |
|---|---|---|---|---|---|---|---|
| A English / salient | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| B English / obscure | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| C Akkadian / obscure | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| D Akkadian / salient | — | — | — | — | — | — | — |

Complete except D.

### Paper dataset ↔ our analog

| Paper dataset | Axis | Our analog |
|---|---|---|
| World / US / NYC place | space | **one** analog: fragment find-spot (lon, lat) |
| Historical figures | time | year (king-name-token flavour, slide 15) |
| Media / art | time | year |
| **News headlines** | time | **year from the whole fragment** ← closest match |

Their 3 space datasets collapse to our 1 geo analog; their 3 time datasets collapse to our
1 year analog.

---

## 3. Where the code and data live

### Two top-level experiment families

**(a) `v_1/src/world_models/` — the G&T replication and its Akkadian mimic** (newer; cells
A, B, C; the four newest slides).

| Piece | Path |
|---|---|
| Shared library | `wm_lib/` — `registry.py` (model specs, `sites`, random-init twins), `entity_data.py` (verbatim ports of the paper's entity-string builders + targets), `extract.py` (`extract_dataset`, last/mean pooling), `probing.py` (`run_probe` ridge, `run_pls_probe`, `score_place`, `score_time`, `sanitize`) |
| English extraction | `extract_acts.py` (`--sites last,mean`) |
| English probe | `probe_wm.py` (`--sites` filter, `--probe ridge|pls`) |
| English PLS-k sweep | `probe_eng_pls.py` (reads best layer from committed ridge JSON) |
| English aggregate | `aggregate_results.py` → the summary CSVs + `results/RESULTS.md` |
| TF-IDF floor | `tfidf_baseline.py` |
| Random-init builder | `build_random_llama.py` |
| Entity data | `data/entity_datasets/*.csv` (the paper's six datasets) |
| **Akkadian sub-package** | `akkadian/` |
| ↳ data loader | `akkadian/akk_data.py` — `TEXT_VARIANTS` (`akk_maximal`, `eng_maximal`, `eng_tier0`), `ruler_set_mask` (r8/r40), `target_values`, `merged_site_labels` (0.1° coord merge → 10 sites), `is_test_split` |
| ↳ MC/LORO modes | `akkadian/akk_modes.py` — `mc_balanced` (by ruler), `mc_site` (by find-spot), `loro` |
| ↳ extraction | `akkadian/extract_akk.py` |
| ↳ probes | `akkadian/probe_akk.py` (holdout/mc/loro), `akkadian/probe_geo_site.py` (by-site geo R²), `akkadian/probe_layers_pls.py` (per-layer + best-layer PLS-k), `akkadian/tfidf_akk.py` |
| Cluster jobs | `sbatch/` (English: `W1*` extract, `W2` probe, `W3` aggregate, `Wm1*`/`Wm2*` mean-pool, `Weng_pls`, `Wgpt_*`) and `akkadian/sbatch/` (`WA1*` extract, `WA2*` probe, `WAg` geo-site, `WAl` layers+PLS, `WAe` encoders, `WAgpt*`) |

**Results (all committed):**

| What | Path |
|---|---|
| English per-layer probe JSONs | `v_1/src/world_models/results/probes/{method}/{dataset}.{site}.ridge.json` |
| English summaries | `results/summary_best_layer_r2.csv`, `summary_best_layer_spearman.csv`, `summary_layerwise.csv` (5130 rows, 15 methods), `RESULTS.md` |
| English PLS-k | `results/eng_pls/{method}/{dataset}.{site}.json` |
| Akkadian probes (holdout/mc/loro) | `akkadian/results/probes/{method}/{variant}.{r8\|r40}.{year\|geo}.{last\|mean}.ridge.json` |
| Akkadian by-site geo | `akkadian/results/probes_geosite/{method}/{variant}.{pool}.geo_site.json` |
| Akkadian layer + PLS-k | `akkadian/results/layers_pls/{method}/{variant}.{target}.{pool}.json` |
| Akkadian roll-ups | `akkadian/results/summary_ALL_modes_full.csv` (132 rows, every metric), `RESULTS_akk_MC_vs_holdout.md` |

Activations (`activations/**/*.npz`) are **cluster-local and gitignored** — only
`metadata.json` is committed. Anything needing raw activations must run on the cluster.

**(b) `v_1/src/stress_tests/` — the older thesis stress tests** (cells B/C; slides 4–23).

| Dir | Experiment | Slides |
|---|---|---|
| `p1_gurnee_tegmark/` | P1 year probe, whole-text mean vs king-name token | 15 |
| `p2_godey_geography/` | P2 find-spot decoding (great-circle km) | 14 |
| `p3_matter_of_time/` | timeline / anchor experiments | (background) |
| `translation/` | translation probe (year + geo) — also holds `translations.parquet` | 17, 18 |
| `e5_shuffle/` | word-order shuffle ablation | 19 |
| `e6_clusters/` | cluster metrics | (background) |
| `p7_ksparse/` | k-sparse probes | (background) |
| `p8_lambda_probe/` | supervision dial | 21 |
| `p9_gkpls/` | geodesic kernel PLS | 20 |
| `p10_reduce_kernels/` | reduce-then-kernel (PCA/PLS/UMAP → kernels) | (not yet in deck) |
| `redo_t9_knowledge/` | T9 free-text date generation | 13 |
| `redo_t10_prompt/` | T10 prompting styles | 16 |
| `t11_gen_dating/`, `t12_forced_dating/` | generation-based dating | 22 |
| `shared/` | `mc_probe.py`, `mc_maxking.py`, `metrics.py` (haversine), `geo_loader.py`, `sites_gazetteer.csv` | — |
| `sbatch/` | `_common.sh` (`sync_main`, `commit_push`) used by *every* job | — |

Results: `v_1/src/stress_tests/results/csv/*.csv` (one per experiment, e.g.
`p2_geo_mc.csv`, `translation_mc.csv`, `p8_lambda.csv`, `tfidf_baseline.csv`) and each
experiment's own `results/` dir. Table builder: `v_1/src/stress_tests/aggregate_tables.py`.

**Corpus / shared data**

| What | Path |
|---|---|
| Akkadian corpus | `v_1/data/evaluation/corpora/orcc_corpus.parquet` (`fragment_id, ruler, year, provenance, text_maximal`) |
| Translations | `v_1/src/stress_tests/translation/translations.parquet` (`eng_tier0` = faithful literal gloss ~1.9k chars; `eng_maximal` = aggressively cleaned, **hallucinates king names — excluded everywhere**) |
| Gazetteer | `v_1/src/stress_tests/shared/sites_gazetteer.csv` |
| Balanced draws | `v_1/src/linear_probing/results/orcc_round2_phase0/balanced_subset/draws_matrix.npy` (and `balanced_subset_sites/` for the by-site version) |

---

## 4. Slide-by-slide map

Columns: **cell** (A/B/C or —), **target**, **pooling**, **probe/metric**, **code**, **role**.

| # | Slide title | Cell | Target | Pool | Probe / metric | Code | Results | Role |
|---|---|---|---|---|---|---|---|---|
| 0 | Title | — | — | — | — | — | — | narrative |
| 1 | The thesis: 400M translation model beats the 120B LLM | — | — | — | — | — | — | narrative |
| 2 | Maximal · mean-pool · balanced · PLS · Spearman (protocol) | — | — | — | — | — | — | narrative |
| 3 | The experimental journey | — | — | — | — | — | — | narrative |
| 4 | PLS vs Ridge across all models — 400M encoder wins | C | time | mean | PLS+ridge, ρ | `stress_tests/` (aggregate) | `results/csv/table1_best_models.csv` | **primary — thesis headline** |
| 5 | Thalesian deepens with layer; LLMs peak mid-network | C | time | mean | PLS, layer | `p1_gurnee_tegmark/` | `results/csv/p1_year_mc.csv` | layer variant |
| 6 | k = 3–5 PLS components capture it | C | time | mean | PLS-k (k≤5) | `p1_gurnee_tegmark/` | `results/csv/p1_year_mc.csv` | **superseded by 32** |
| 7 | Scale and next-token finetuning both do nothing | C | time | mean | ridge | `stress_tests/` | `results/csv/table1_best_models.csv` | ablation |
| 8 | Thalesian wins despite worst Akkadian tokenizer | C | — | — | tokenizer stats | `stress_tests/eda/` | — | analysis |
| 9 | Translation finetune builds a deep representation | C | time | mean | PLS, layer | `p1_gurnee_tegmark/` | `results/csv/p1_year_mc.csv` | layer variant |
| 10 | Chronology entangled in Qwen's embedding space | C | time | mean | PCA/UMAP viz | `e6_clusters/` | `results/csv/e6_cluster_indices.csv` | geometry viz |
| 11 | Contributions and gaps | — | — | — | — | — | — | narrative |
| 12 | Stress-testing the linear-timeline claim (intro) | — | — | — | — | — | — | narrative |
| 13 | T9 — do models know the dates? (free-text) | C | time | — | generation | `redo_t9_knowledge/` | `results/csv/t9_knowledge.csv` | non-probe |
| 14 | P2 — decode the find-spot from whole-text embedding | B+C | space | mean | PLS+ridge, **km** | `p2_godey_geography/probe_p2_mc.py` | `results/csv/p2_geo_mc.csv` | **primary (km)** |
| 15 | P1 — year: whole-text mean vs king-name token | C | time | **mean vs entity-token** | ridge, ρ | `p1_gurnee_tegmark/` | `results/csv/p1_maxking.csv` | **pooling variant** |
| 16 | T10 — does prompting help? | C | time | — | prompt styles | `redo_t10_prompt/` | `results/csv/t10_mc.csv` | non-probe |
| 17 | Translation probe — Year | B (vs C ref) | time | mean | PLS+ridge, ρ | `translation/probe_translation_mc.py` | `results/csv/translation_mc.csv` | **primary (B, time)** |
| 18 | Translation probe — geo | B (vs C ref) | space | mean | PLS+ridge, **km** | `translation/probe_translation_mc.py` | `results/csv/translation_mc.csv` | **≈ duplicate of 14** |
| 19 | E5 — shuffle the words | C | time | mean | ridge, ρ | `e5_shuffle/probe_e5_mc.py` | `results/csv/e5_shuffle.csv` | order ablation |
| 20 | P9 — geodesic kernel PLS | C | time | mean | kernel PLS | `p9_gkpls/` | `results/csv/p9_gkpls.csv` | probe variant |
| 21 | P8 — the supervision dial | C | time | mean | λ dial | `p8_lambda_probe/` | `results/csv/p8_lambda.csv` | probe variant |
| 22 | T12 — ask the LLM directly | C | time | — | generation | `t12_forced_dating/` | `results/csv/t12_forced_dating.csv` | non-probe |
| 23 | The tier0 baseline vs dumb controls | B | both | mean | ridge | `stress_tests/` | `results/csv/tfidf_baseline.csv` | control |
| 24 | **Our models do represent space & time — on English** | **A** | both | last | ridge, R² | `world_models/probe_wm.py` | `results/summary_best_layer_r2.csv` | **primary (A)** |
| 25 | Year — from the faithful English gloss | B | time | last | ridge, **balanced-MC r8**, R²+ρ | `akkadian/probe_akk.py` | `akkadian/results/probes/*/eng_tier0.r8.year.last.*` | **primary (B, time)** |
| 26 | Geo — find-spot R² from the English gloss | B | space | last+mean | ridge, **by-site MC**, R² | `akkadian/probe_geo_site.py` | `akkadian/results/probes_geosite/*/eng_tier0.*` | **primary (B, space)** |
| 27 | Year — from the raw Akkadian | C | time | last | ridge, **balanced-MC r8**, R²+ρ | `akkadian/probe_akk.py` | `akkadian/results/probes/*/akk_maximal.r8.year.last.*` | **primary (C, time)** |
| 28 | Geo — find-spot R² from the raw Akkadian | C | space | last+mean | ridge, **by-site MC**, R² | `akkadian/probe_geo_site.py` | `akkadian/results/probes_geosite/*/akk_maximal.*` | **primary (C, space)** |
| 29 | English — where in the network do space & time live? | A | both | last+mean | ridge, layer | `probe_wm.py` + `aggregate_results.py` | `results/summary_layerwise.csv` | **primary (A, depth)** |
| 30 | English — how many PLS components? | A | both | last+mean | PLS-k 1…64 | `probe_eng_pls.py` | `results/eng_pls/` | **primary (A, dimensionality)** |
| 31 | Akkadian — the same layer analysis | B+C | both | last+mean | ridge, layer | `akkadian/probe_layers_pls.py` | `akkadian/results/layers_pls/` | **primary (B/C, depth)** |
| 32 | Akkadian — PLS components at the best layer | B+C | both | last+mean | PLS-k 1…64 | `akkadian/probe_layers_pls.py` | `akkadian/results/layers_pls/` | **primary (B/C, dimensionality)** |

---

## 5. Model ladder and controls (consistent across all cells)

Trained: `llama2_7b`, `llama2_13b`, `llama2_70b` (the paper's ladder), `qwen3_1b7`,
`qwen3_8b`, `qwen3_32b`, `gpt_oss_120b`, and three small encoders
(`thalesian_akk300m`, `thalesian_cunei400m`, `umt5_base`).
Controls: **random-init twins** `llama2_7b_random`, `llama2_13b_random`,
`llama2_70b_random`, `random` (random-init Qwen3-8B), plus a **TF-IDF** floor.

Reading rule used throughout: *a score witnesses learning only if it beats **both** the
TF-IDF floor **and** the arm's own random-init twin.*

Plot conventions in the four newest slides: Qwen = blues, Llama = greens, encoders =
orange/purple, darker = larger; dashed = random controls; ★ = the maximum of each curve;
geo panels use a symlog y-axis so the negative tail compresses.

---

## 6. Headline numbers a story can lean on

* **A (English)** — Llama-2-70B reproduces the paper within ~0.02 R² on all six datasets
  (world .905 vs .911). Trained vs its random twin: **.905 vs .170**.
* **gpt-oss-120B lands mid-pack** (world .807), below the Llama ladder despite being the
  largest — scale alone does not buy a better world model.
* **B (English gloss of Akkadian)** — signal drops but survives; TF-IDF starts to lead on
  time (year r8 MC: TF-IDF ρ .775 vs best trained ρ .557).
* **C (raw Akkadian)** — trained arms sit **at or below** their random twins on year
  (e.g. Llama-2-7B ρ .433 vs its random .438); TF-IDF ρ .707 dominates.
* **Geo behaves differently from time** — under by-site R², trained models *do* beat the
  TF-IDF floor (~.25–.34 vs .02) on `last` pooling. Space survives where time collapses.
* **Under LORO (unseen ruler)** the year signal collapses to ≈0 for every arm — the
  holdout number was ruler-identity memorisation.
* **PLS-k** — trained arms keep gaining to k ≈ 8–16; random arms saturate by k ≈ 3–5.
* **`eng_maximal` is broken** (cleaner hallucinates king names) and is excluded
  everywhere; `eng_tier0` is the valid English variant.

---

## 7. Known redundancy to resolve when reordering

1. **18 ≈ 14** — same P2 by-site km protocol; 18 only swaps the text variant and repeats
   14's Akkadian column as its reference. *Strongest cut candidate.*
2. **6 → 32** — the old k ≤ 5 sweep is superseded by k ≤ 64 across all arms.
3. **14/18 (km) vs 26/28 (R²)** — same probe, different read-out. Either keep km as the
   interpretable-distance framing, or fold km in as a secondary column on 26/28 and cut.
4. **17 vs 25** — both ask "does English surface the year?", but 17 is holdout+PLS+mean and
   25 is balanced-MC+ridge+last. Genuinely different protocol; 25 is the honest one.
5. **P10 (`p10_reduce_kernels/`) has results but no slide** — reduce-then-kernel; decide
   whether it earns one.

## 8. Story spine the matrix supports

1. **A** — the paper reproduces on English on our ladder, and the random-init controls
   (which the paper never ran) show the geometry is learned, not architectural.
2. **A → B** — keep English, swap famous entities for obscure ones: signal weakens, the
   dumb n-gram floor starts winning on time.
3. **B → C** — keep the entities, swap English for Akkadian: trained collapses onto its
   random twin.
4. **Conclusion** — the linear world model is a property of the **language the model was
   trained on**, not of scale; and our 400M translation encoder, not the 120B LLM, is the
   system that actually dates Akkadian.

## 9. Practical notes for whoever edits next

* Cluster jobs all `source v_1/src/stress_tests/sbatch/_common.sh` and use `sync_main` /
  `commit_push`, so **results land on `main` by themselves**. Check with `git log`.
* Activations are gitignored; anything needing them runs on the cluster. Probe JSONs are
  committed, so **aggregation and plotting can be re-run locally**.
* fp16 activations can overflow to ±inf (gpt-oss especially) — always route through
  `probing.sanitize()` before fitting.
* Regenerating the four newest figures needs only committed JSON/CSV; no cluster.
