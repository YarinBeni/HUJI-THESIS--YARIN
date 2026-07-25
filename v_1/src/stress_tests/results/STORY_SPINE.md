# Story spine v2 — "The ladder" (Option 1 + matrix as the map)

Agreed narrative for `thesis_story_9.html`, replacing the chronological three-era order.
Framing: *a researcher's follow-up to Gurnee & Tegmark (2023), ICLR-style.*
We reproduce the paper, then climb away from its comfort zone one factor at a time
(famous→obscure entities, entity→fragment, English→Akkadian), using the 2×2 matrix
{salient, obscure} × {high, low resource} as the recurring visual map. We close by
stating the conditions a temporal/spatial world model needs to emerge, and show that a
translation objective supplies them where pretraining scale cannot.

Slide counter: **29 slides** (new indices 0–28). Cut: old 3, 6, 10, 12, 18 (see bottom).

## Master mapping — new order

| New | Old | Act | Experiment | Exact config | Data source | Status |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 Motivation | — | title | — | retitle to follow-up framing |
| 1 | 1 | 0 | — | motivation: world models → archaeology; teaser of inversion | — | **rewrite** |
| 2 | new (from 12) | 0 | — | G&T claim + their single-cell setting (salient · English · last · ridge · R²) | paper | **new** |
| 3 | new | 0 | — | the 2×2 matrix as climbing map; D empty; recurring progress marker | — | **new** |
| 4 | 2 | 1 Cell A | — | protocol & controls: ladder, random twins, TF-IDF floor, reading rule | — | **rework** (must cover BOTH protocol families, see issue #1) |
| 5 | 24 | 1 | G&T replication | 6 datasets · last · ridge · R² · random twins · gpt-oss | `world_models/results/summary_best_layer_r2.csv` | keep |
| 6 | 29 | 1 | English layer sweep | last+mean · ridge · per-layer | `summary_layerwise.csv` | keep |
| 7 | 30 | 1 | English PLS-k | best layer · k=1..64 | `results/eng_pls/` | keep |
| 8 | 25 | 2 Cell B | Akk year, English gloss | eng_tier0 · r8 · balanced-MC · last · ridge · R²+ρ | `akkadian/results/probes/*/eng_tier0.r8.year.last.*` | keep; add matrix marker → B |
| 9 | 26 | 2 | Akk geo, English gloss | eng_tier0 · by-site MC · last+mean · ridge · R² | `akkadian/results/probes_geosite/*/eng_tier0.*` | keep |
| 10 | 23 | 2 | tier0 vs dumb controls | eng_tier0 · mean · ridge vs TF-IDF/random | `results/csv/tfidf_baseline.csv` | keep |
| 11 | 17 | 2 | translation probe · year | eng_tier0 · holdout · mean · PLS+ridge · ρ | `results/csv/translation_mc.csv` | demote to "supporting"; reconcile protocol wording vs new 8 |
| 12 | 27 | 3 Cell C | Akk year, raw Akkadian | akk_maximal · r8 · balanced-MC · last · ridge · R²+ρ | `akkadian/results/probes/*/akk_maximal.r8.year.last.*` | keep; matrix marker → C |
| 13 | 28 | 3 | Akk geo, raw Akkadian | akk_maximal · by-site MC · last+mean · ridge · R² | `akkadian/results/probes_geosite/*/akk_maximal.*` | keep |
| 14 | 14 | 3 | P2 find-spot in km | akk_maximal · mean · PLS+ridge · great-circle km | `results/csv/p2_geo_mc.csv` | keep as interpretable-km companion of new 13 |
| 15 | 31 | 3 | Akkadian layer sweep | both variants · last+mean · ridge · per-layer | `akkadian/results/layers_pls/` | keep |
| 16 | 32 | 3 | Akkadian PLS-k | best layer · k=1..64 | `akkadian/results/layers_pls/` | keep (supersedes old 6) |
| 17 | 15 | 3 | P1 pooling/LORO | whole-text mean vs king-token · ridge · ρ · LORO | `results/csv/p1_maxking.csv` | keep; leads into rescue-attempts act |
| 18 | 13 | 4 Rescue | T9 knowledge | free-text generation | `results/csv/t9_knowledge.csv` | keep |
| 19 | 22 | 4 | T12 forced dating | forced behavioral answer | `results/csv/t12_forced_dating.csv` | keep |
| 20 | 16 | 4 | T10 prompting | prompt styles · MC | `results/csv/t10_mc.csv` | keep |
| 21 | 19 | 4 | E5 shuffle | word-order scramble · mean · ridge · ρ | `results/csv/e5_shuffle.csv` | keep |
| 22 | 21 | 4 | P8 supervision dial | λ dial | `results/csv/p8_lambda.csv` | keep |
| 23 | 20 | 4 | P9 geodesic KPLS | kernel PLS | `results/csv/p9_gkpls.csv` | keep |
| 24 | new | 5 Conditions | — | synthesis: conditions for a world model (language resource + salience; not scale/prompting/probe power) | acts 1–4 | **new** |
| 25 | 4 | 5 | thesis headline | akk_maximal · mean · balanced · PLS+ridge · ρ | `results/csv/table1_best_models.csv` | keep |
| 26 | 5+9+7 | 5 | why translation works | layer profile Thalesian vs uMT5 vs Qwen; scale/NTP-finetune null | `results/csv/p1_year_mc.csv`, `table1_best_models.csv` | **merge 3→1..2 slides** |
| 27 | 8 | 5 | tokenizer autopsy | tok/word stats | eda | keep |
| 28 | 11 | 5 | contributions | — | — | **rework** to follow-up-paper framing |

## Cut / appendix

| Old | Fate | Reason |
|---|---|---|
| 3 (journey) | appendix | process, not argument |
| 6 (k≤5 PLS) | cut | superseded by old 32 (new 16) |
| 10 (geometry viz) | appendix | weakest link in ladder arc; PCA/UMAP caveats |
| 12 (stress-test intro) | absorbed into new 2–3 | framing now global |
| 18 (translation geo) | cut | duplicate of old 14 protocol |

## Open issues found in step 2

1. **Two protocol families coexist and the protocol slide describes only one.**
   Old-thesis slides (25–27 new) use *maximal · mean · balanced-MC · PLS · Spearman*;
   world-models slides (5–16 new) use *last (year) / last+mean (geo) · ridge · R²+ρ ·
   balanced-MC r8 / by-site MC*. New slide 4 must present both and say why they differ
   (paper-faithful protocol vs honest-corpus protocol), or numbers will look contradictory.
2. **Layer-story duplication risk:** new 26 (p1_year_mc PLS·mean layer curves) vs new 15
   (layers_pls ridge·last/mean). Same question, different pipelines. Keep both but the
   takeaways must cite their own configs explicitly.
3. **17-vs-25 (old) reconciled** by demoting old 17 to "supporting" in Act 2 (new 11).
4. **P10 (reduce-then-kernel)** still has no slide; candidate for Act 4 as a 7th rescue
   attempt or appendix. Undecided.
5. New builds required: matrix map (new 3), conditions synthesis (new 24), paper-intro
   (new 2), plus rewrites of motivation (1), protocol (4), contributions (28).
