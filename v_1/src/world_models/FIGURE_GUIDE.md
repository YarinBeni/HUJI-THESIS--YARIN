# Figure guide — every plot, what it shows, where it lives, what made it

Written to be used while assembling the deck. For each figure: the **exact repo path**,
the **script that regenerates it**, what it shows, and where it came from (replicating a
paper vs. our own extension).

All paths below are relative to the repo root, `HUJI-THESIS--YARIN/`.

---

## 0. Read this first — there are TWO figure families

This is the thing that is easy to lose track of.

| family | directory | made by | status |
|---|---|---|---|
| **DECK figures** | `v_1/src/world_models/results/figs/` | `v_1/src/world_models/plot_*.py` | **already embedded** in `thesis_story_9.html` as base64 |
| **ANALYSIS figures** | `v_1/src/world_models/figures/` and `figures/designs/` | `figures/make_curves.py`, `figures/designs/*.py`, `manifold/make_isometry_summary.py` | **not in the deck yet** — these are the ones to choose from |

Both families now use the **same palette and type scale** (`figures/lib/_style.py` is
slaved to `plot_cellA_figs.py`), so they can sit on adjacent slides without a colour
meaning two different things.

**Colour code, both families:**
BLUES = Qwen3 + gpt-oss (light→dark with size) · GREENS = Llama-2 (light→dark with size)
· WARM = the three translation encoders · **PURPLE = random-init controls** (always
dashed) · BLACK = TF-IDF floor (dotted).

**Every figure exists twice**: `name.png` (300 dpi, for the HTML deck) and `name.pdf`
(vector, for the thesis PDF — use this one in LaTeX).

### The four cells, referenced everywhere

| cell | stimulus | salience | resource |
|---|---|---|---|
| **A** | famous English entities ("George Washington", world places) | salient | high |
| **B** | Assyrian ruler names, in English ("Ashurbanipal") | obscure | high |
| **B′** | whole fragments, faithful English gloss (tier-0) | obscure | glossed |
| **C** | whole fragments, raw Akkadian transliteration | obscure | low |

### The three read-outs

Each design renders three times from the same code, different table:

| suffix | probe used | table |
|---|---|---|
| `name.png` | ridge everywhere | `figures/TIDY_all_year_results.csv` |
| `name__deck.png` | **what the thesis reports** — PLS on fragments, PLS-5 on obscure entities, ridge on A | `figures/TIDY_all_year_results__mc_group__deck.csv` |
| `name__pls.png` | best-k PLS wherever a sweep exists | `figures/TIDY_all_year_results__mc_group__pls.csv` |

**Use `__deck` for anything compared against the thesis.**

---

## 1. DECK figures — already in the HTML

These came from the deck branch and are what the current slides show.

### `results/figs/fig_cellA_layers.png` · `fig_cellA_plsk.png`
**Script:** `v_1/src/world_models/plot_cellA_figs.py`
**Provenance:** direct replication of **Gurnee & Tegmark 2023**, *Language Models
Represent Space and Time*.
2×4 panels: rows = SPACE (World/USA/NYC → lat,lon) and TIME (Figures/Art/Headlines →
year); columns = last-pooling then mean-pooling, R² then Spearman within each. Shows all
four read-outs the paper mentions, so nothing is a deviation from it. **This is the
"our harness reproduces the paper" evidence.**

### `results/figs/fig_cellC_layers.png` · `fig_cellC_plsk.png`
**Script:** `plot_cellC_figs.py` — the same layout applied to **our** Akkadian corpus.
Directly comparable to the cell-A pair above; that comparability is the point.

### `results/figs/fig_encoders_translation.png`
**Script:** `plot_encoders_fig.py` — the three translation encoders vs the LLM ladder.
Ours, not from a paper.

### `results/figs/fig_finetune_ntp.png`
**Script:** `plot_finetune_fig.py` — next-token finetuning on our Akkadian changes
nothing. Ours.

### `results/figs/fig_mlm_arch.png`
**Script:** `plot_mlm_arch.py` — Ithaca-style architecture diagram for the MLM slide.
Schematic, no data.

---

## 2. ANALYSIS figures — the seven designs

All in `v_1/src/world_models/figures/designs/`. All ours; none replicate a paper figure.

### `designs/slopegraph.png` (+ `__deck`, `__pls`)
**Script:** `figures/designs/slopegraph.py`
**The lead figure.** Four stages left→right (A → B → B′ → C). y = **Δ Spearman ρ against
each arm's own random-init twin in the identical configuration** — not raw ρ, because
the protocols differ per stage so absolute ρ is not comparable across stages, while
"how far past your own untrained self" is. Marker shape = which pooling won. Dashed grey
= random-init Llama twins. Black dotted = TF-IDF.
**Shows:** the gap decays monotonically **+.254 → +.217 → +.141 → +.027**, and TF-IDF
falls to −.30 on raw Akkadian.

### `designs/heatmap.png` (no variants)
**Script:** `figures/designs/heatmap.py`
Rows = 15 arms, columns = cell × pooling. **RIDGE panel left, PLS panel right.** Cell
text = raw ρ; cell colour = ρ minus the random-init Qwen reference in the same column;
black frame = best arm per column.
**Shows:** the whole result set at once, and the one place you can see the two probes
disagree — TF-IDF is top of the ridge panel and bottom of the PLS panel. No `__deck` /
`__pls` twin because this figure *is* that comparison.

### `designs/ridgeline.png` (+ `__deck`, `__pls`)
**Script:** `figures/designs/ridgeline.py`
One ridge per configuration (10 total; entity above the labelled CLIFF divider, fragment
below). Each ridge is a **discrete 0.05-wide histogram** over the 14 arms — one bar-step
= one arm, so heights compare across ridges. Per-arm dots on the baseline; TF-IDF is the
black diamond traced by a dotted line.
**Shows:** the fragment cells aren't merely lower, they're *compressed* — every arm piles
into one narrow band.

### `designs/dumbbell.png` (+ `__deck`, `__pls`)
**Script:** `figures/designs/dumbbell.py`
One row per arm × configuration. Filled dot = trained, open dot = its matched random-init
twin, connecting bar **green when trained wins, red when it loses**.
**Shows:** the cliff at per-arm granularity. All the red bars are in the fragment cells,
including Llama-2-70B-random beating several trained arms on raw Akkadian.

### `designs/anatomy.png` (+ `__deck`, `__pls`)
**Script:** `figures/designs/anatomy.py`
The explainer. Literal stimulus text per cell, pooled span underlined, arrow onto the
exact token the probe reads, next to the score.
**Use this before any results slide** — it pre-empts "wait, what is it being shown?".

### `designs/geometry.png` (+ `__deck`, `__pls`)
**Script:** `figures/designs/geometry.py`
2×4 embeddings at each cell's best layer, coloured by chronological rank. **Top row =
supervised PLS-2D, bottom row = unsupervised UMAP.**
**Shows:** at entity level both rows show a time gradient; at fragment level PLS can
still forge a weak axis (it is told the answer) while UMAP shows none. The gap between
the rows is the visual statement of the cliff.

### `designs/kprofile.png` (new)
**Script:** `figures/designs/kprofile.py`
ρ vs PLS rank k, per fragment cell × pooling. Ring = the selected k; warm dashed vertical
= the **k = 5 ceiling** inherited from `stress_tests/shared/mc_probe.py`.
**Shows:** the curves are **flat past k≈5**, so the truncated grid was a real hole but
changed nothing (median gain from widening to 64: **+0.000**). Reads `pls_per_k` directly
from the probe JSONs, so it has no `__deck`/`__pls` twin.

---

## 3. ANALYSIS figures — layer sweeps and PLS curves

All in `v_1/src/world_models/figures/`. **Script for all seven:**
`figures/make_curves.py --which all`

| path | shows |
|---|---|
| `figures/02_cellA_layers.png` | ρ vs layer depth, cell A. Salient English entities peak **late** (Llama-2-70B ≈ layer 53) — the Gurnee–Tegmark profile. |
| `figures/03_cellA_pls.png` | ρ vs PLS components, cell A. Saturates after very few — a salient entity's date is a **low-dimensional, nearly linear** direction. |
| `figures/04_cellB_entity_layers.png` | same sweep for obscure ruler names. Peak lower and **earlier** than cell A. |
| `figures/05_cellB_entity_pls_bare.png` | PLS curve, name alone ("Ashurbanipal") — the paper-faithful probe. |
| `figures/06_cellB_entity_pls_all.png` | PLS curve, name inside a carrier sentence. Compare with 05: context adds little, so the signal is in the **name token**. |
| `figures/07_fragment_layers.png` | layer sweep for whole fragments. Flat and low at every depth — no layer where document chronology appears. |
| `figures/08_fragment_pls.png` | PLS curve for fragments. Contrast with 03: no clean saturation. |

Two more in the same directory:

| path | script | shows |
|---|---|---|
| `figures/00_MASTER_gap_by_stimulus_pooling.png` | `figures/make_master_fig.py` | the gap ladder as a single master panel |
| `figures/01_cross_levels.png` | `figures/make_curves.py` | all four cells side by side |

---

## 4. ANALYSIS figures — embedding panels

All in `v_1/src/world_models/figures/`, built from `figures/lib/six_panel.py`.

| path | shows |
|---|---|
| `figures/09_cellA_sixpanel_PLS.png` / `10_cellA_sixpanel_UMAP.png` | six arms, cell A. Gradient obvious in PLS and **survives in UMAP** → the structure is intrinsic, not an artifact of supervision. |
| `figures/11_cellB_rulers_sixpanel_PLS.png` / `12_..._UMAP.png` | Assyrian ruler names. Weaker in PLS, noticeably noisier in UMAP — the honest picture of an obscure entity. |
| `figures/13_cellB_places_sixpanel_PLS.png` | Mesopotamian place names — the *space* analog. |
| `figures/14_cellA_worldplace_sixpanel_PLS.png` | world places. **The strongest geometry anywhere in the study** — the reference for what a real learned coordinate looks like. |

---

## 5. ANALYSIS figures — manifold geometry

**Provenance:** diagnostics from **Modell et al., arXiv 2505.18235** and the reducibility
indices from **Engels et al.**, run on **our** activations. Method from the papers, data
ours.

### `figures/19_isometry_summary.png` ← **the one for the deck**
**Script:** `v_1/src/world_models/manifold/make_isometry_summary.py`
Condenses all **332 full-activation runs**. Two panels:
- **ρ** = Pearson(kNN-graph geodesic distance, feature distance)
- **ξ** = Chatterjee(squared feature distance, cosine similarity)

Filled dot = trained arm, open dot below = its random-init twin, bar = the gap.
**Shows — this is the finding:** **ρ separates trained from random** (world places .290
vs .101), **ξ does not** (fragment level: Llama-2-70B .556 vs its own twin .463; an
untrained Qwen on the English gloss reaches .531).
**So do not quote ξ as evidence of a world model.** Quote ρ, against the matched control.

### `figures/17_arc__akk__llama2_13b__akk_maximal__year__mean.png`
**Script:** `manifold/manifold_figs.py --method llama2_13b --surface akk`
Single-cell detail: PCA pairs of the L2-normalised, rank-4-denoised representation,
coloured by year. Four pairs are drawn because in their paper the structure often lives
in **PC3×PC4**, not PC1×PC2.

### `figures/18_isometry__akk__llama2_13b__akk_maximal__year__mean.png`
Same script. The two diagnostic scatters for that cell — cosine vs squared Δyear (ξ
annotated) and graph-geodesic vs Δyear (ρ annotated). **Show this before 19** so the
audience knows how ξ and ρ are computed.

### `figures/15_reducibility_indices.png`
**Script:** `manifold/reducibility.py` → `figures/lib/modell_figs.py`
Engels et al. reducibility: the **ε-mixture index M** (is this direction just a mixture
of independent 1-D features?) and the **separability index S** (minimum mutual
information over rotations). Low M + low S = a genuinely irreducible multi-dimensional
feature rather than two stacked scalars.

### `figures/16_year_metric_choice.png`
Same script. Sanity check on the feature metric: absolute Δyear vs their log-recency
reparameterisation. Confirms the isometry conclusions aren't an artifact of how
"temporal distance" was defined.

### The full set
`v_1/src/world_models/manifold/figs/` holds all **664 PNGs** (332 runs × arc + isometry)
plus a `*__stats.json` per run. `19_isometry_summary.png` is built from those JSONs.

---

## 6. Provenance at a glance

| origin | figures |
|---|---|
| **Replicating Gurnee & Tegmark** | `results/figs/fig_cellA_layers`, `fig_cellA_plsk`; analysis-side `02`, `03`, `09`, `10`, `14` |
| **Same method, our Akkadian corpus** (deliberately comparable to the above) | `results/figs/fig_cellC_layers`, `fig_cellC_plsk`; `04`–`08`, `11`–`13` |
| **Method from Modell / Engels, our data** | `15`, `16`, `17`, `18`, `19` |
| **Entirely ours** | all seven `designs/*`; `00`, `01`; `results/figs/fig_encoders_translation`, `fig_finetune_ntp` |

---

## 7. Regenerating

```bash
cd v_1/src/world_models/figures
python3 build_tidy.py --mode mc_group --readout deck     # and: raw | pls

cd designs
for f in ridgeline slopegraph heatmap dumbbell anatomy geometry kprofile; do
  python3 $f.py                                                                # ridge
  TIDY_CSV=../TIDY_all_year_results__mc_group__deck.csv FIG_TAG=__deck python3 $f.py
  TIDY_CSV=../TIDY_all_year_results__mc_group__pls.csv  FIG_TAG=__pls  python3 $f.py
done

cd .. && python3 make_curves.py --which all && python3 make_master_fig.py
cd ../manifold && python3 make_isometry_summary.py
cd .. && python3 plot_cellA_figs.py && python3 plot_cellC_figs.py   # deck family
```

Resolution knobs (`figures/lib/_save.py`): `FIG_DPI=450` for a poster, `FIG_PDF=0` to
skip the vector copy.

---

## 8. Caveats to keep attached

1. **7 of 58 fragment cells** still carry the old 4-point PLS grid, after a truncated-write
   corruption in the WAk job. Re-run pending; the ordering does not change.
2. **`pls_best_k` is selected on the outer test folds** — the deck's convention, but
   optimistic by a median of **0.029**. `pls_nested_spearman_mean` is the honest number.
3. **R² is not reported for fragment cells.** Under GroupKFold a test fold is essentially
   one ruler, so year variance ≈ 0 and R² degenerates to −0.22 for every arm including
   the floor.
4. **Cell C is weak, not clean, evidence** — Llama-2-70B-random scores .322 there, third
   overall.
5. **`WBk` (cell B k-sweep) has not finished**, so `04`–`06` still use fixed PLS-5.
