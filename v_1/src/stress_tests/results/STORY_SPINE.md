# Story spine v3 — "The ladder" (Option 1 + matrix as the map)

Agreed narrative for `thesis_story_9.html`. Framing: *a researcher's follow-up to
Gurnee & Tegmark (2023), ICLR-style.* We reproduce the paper, then climb away from its
comfort zone one factor at a time, using the 2×2 matrix {salient, obscure} × {high, low
resource} as the recurring map. Within each cell we move entity → fragment, mirroring the
paper's entity probes before generalising. Controls (random twins, TF-IDF) are **rows in
every results table, never dedicated slides**. Metrics: R² like the paper (+ Spearman for
year). The km read-out is dropped.

~28 slides. v3 changes vs v2: Act 2/3 split entity-first-then-fragment; controls slide cut;
old 17 cut (pipeline-duplicate of the world-models B result); km slide cut; confounder
slide added; ruler≠chronology slide added; T12+T10 merged; finetune null moved to rescue
act; Act-5 objective slide now uMT5 vs AKK-300M vs cunei-400M.

## Master mapping — new order

| New | Old | Act | Content | Config / evidence | Data | Status |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | Title | — | — | retitle |
| 1 | 1 | 0 | Motivation: world models → archaeology; extending G&T's *headlines* experiment (their only fragment-like dataset) to obscure entities, a low-resource language, and mean pooling. Teaser of the inversion. | — | — | **rewrite** |
| 2 | new(12) | 0 | The paper: G&T claim + their single cell (salient · English · entity string · last token · ridge · R²) | — | paper | **new** |
| 3 | new | 0 | The matrix — climbing map; D empty; returns as progress marker | — | — | **new** |
| 4 | 2 | 1 | Protocol: ladder, both pipelines, reading rule (beat TF-IDF *and* your random twin) | — | — | **rework** |
| 5 | 24 | 1 | Cell A repro: paper reproduces on our ladder; random twins flat; gpt-oss mid-pack | 6 datasets · last · ridge · R² | `world_models/results/summary_best_layer_r2.csv` | keep |
| 6 | 29 | 1 | A: layer sweep | last+mean · ridge | `summary_layerwise.csv` | keep |
| 7 | 30 | 1 | A: PLS-k | best layer · k≤64 | `results/eng_pls/` | keep |
| 8 | — | 2 | **B-entity: obscure entities, paper pooling.** Ruler names written in English, last token, year. Strict A→B step: only salience changes. | entity string · last · ridge | **MISSING RUN** | **placeholder — needs small cluster job** (geo target at entity level undefined; year only unless decided otherwise) |
| 9 | 25 | 2 | B-fragment: year from English gloss, last **and** mean | eng_tier0 · r8 · MC · ridge · R²+ρ | `akkadian/results/probes/*/eng_tier0.r8.year.*` | keep; show both poolings |
| 10 | 26 | 2 | B-fragment: geo from English gloss | eng_tier0 · by-site MC · last+mean · R² | `akkadian/results/probes_geosite/*/eng_tier0.*` | keep |
| 11 | new | 3 | **The confounder control** (Act-3 opener): maximal cleaning + balanced MC. Without them TF-IDF wins by majority-guessing king-name surface strings — label leakage, not language understanding. | protocol rationale + before/after numbers | `tfidf_baseline.csv`, `summary_ALL_modes_full.csv` (hold vs mc) | **new** |
| 12 | 15 | 3 | C-entity: king-name token in the Akkadian text (paper-style entity probe, our language) | king_last vs king_mean vs whole-text mean | `results/csv/p1_maxking.csv` | rework framing |
| 13 | 27 | 3 | C-fragment: year from raw Akkadian — trained ≈ random twin | akk_maximal · r8 · MC · last+mean · R²+ρ | `akkadian/results/probes/*/akk_maximal.r8.year.*` | keep |
| 14 | 28 | 3 | C-fragment: geo from raw Akkadian — space partially survives | akk_maximal · by-site MC · last+mean · R² | `akkadian/results/probes_geosite/*/akk_maximal.*` | keep |
| 15 | 31 | 3 | C: layer sweep mirror | last+mean · ridge | `akkadian/results/layers_pls/` | keep |
| 16 | 32 | 3 | C: PLS-k mirror | best layer · k≤64 | `akkadian/results/layers_pls/` | keep |
| 17 | new | 3 | **Ruler ≠ chronology**: king_last ruler-F1 .98, stratified ρ .98, but group-level ρ ≈ 0; LORO collapses to ≈0 | maxking + LORO | `p1_maxking.csv`, `summary_ALL_modes_full.csv` (loro) | **new** (assembled from existing data) |
| 18 | 13 | 4 | Rescue 1 — T9: does it *know* the dates? (free text) | generation | `t9_knowledge.csv` | keep |
| 19 | 22+16 | 4 | Rescue 2 — ask it directly: T12 forced dating + T10 prompting, **one table** | generation + prompt styles | `t12_forced_dating.csv`, `t10_mc.csv` | **merge** |
| 20 | 7 | 4 | Rescue 3 — scale & NTP finetuning on all our Akkadian: null at every scale | +NTP bars | `table1_best_models.csv` | move here from old thesis block |
| 21 | 19 | 4 | Rescue 4 — E5 shuffle: is it word order? | scramble · mean · ridge | `e5_shuffle.csv` | keep |
| 22 | 20 | 4 | Rescue 5 — non-linear probes: P9 geodesic kernels | kernel PLS | `p9_gkpls.csv` | keep (merge with 23 in step 3 if tables are small) |
| 23 | 21 | 4 | Rescue 6 — P8 supervision dial | λ dial | `p8_lambda.csv` | keep |
| 24 | new | 5 | **Conditions for a world model** (synthesis): high-resource language + salient entities; scale, prompting, supervision, probe power don't substitute | acts 1–4 | — | **new** |
| 25 | 4 | 5 | What does work: 400M translation encoder beats the 120B LLM | akk_maximal · mean · MC · PLS+ridge · ρ | `table1_best_models.csv` | keep |
| 26 | 9(+5) | 5 | Why: the translation objective. uMT5-base (vanilla) vs AKK-300M (Akkadian-only) vs cunei-400M (multilingual cuneiform) — multilingual translation finetune builds the deep signal | layer profiles, same-size comparison | `akkadian/results/layers_pls/{umt5_base,thalesian_akk300m,thalesian_cunei400m}` | **rebuild with better plot (step 3)** |
| 27 | 8 | 5 | Despite the worst tokenizer | tok/word | eda | keep |
| 28 | 11 | 5 | Contributions & conclusion, follow-up-paper framing | — | — | **rework** |

## Cut / appendix

| Old | Fate | Reason |
|---|---|---|
| 3 (journey) | appendix | process, not argument |
| 6 (k≤5 PLS) | cut | superseded by new 16 |
| 10 (geometry viz) | appendix | weakest link in ladder arc |
| 12 (stress intro) | absorbed | framing now global (new 2–3) |
| 14 (P2 km) | cut | metric standardised to R²/ρ like the paper |
| 17 (translation-probe year) | appendix | pipeline-duplicate of new 9 (holdout+PLS+mean vs MC+ridge+last answers the same question with different numbers) |
| 18 (translation-probe geo) | cut | duplicate of old 14's protocol |
| 23 (tier0 baseline) | cut | controls are rows in every table, not a slide |

## Open items

1. **B-entity English run missing** (new 8): small cluster job — build ruler-name entity
   CSV (English spellings + reign midpoint year), reuse `extract_acts.py`/`probe_wm.py`.
   Until then the slide is an explicit placeholder. Geo target at entity level undefined.
2. P10 (reduce-then-kernel) still slide-less; candidate extra rescue slide or appendix.
3. Step 3 will replace table slides with plots where a figure tells it better; slide 26
   explicitly flagged for a better plot.
