# Figure guide — one sharp paragraph per plot

Caption text here is written to be pasted straight into the HTML deck. Every figure
exists as a 300 dpi PNG (deck) and a vector PDF (thesis PDF). Regenerate anything with
the command in its row.

**The four cells referenced throughout** — this is the salience × resource matrix:

| cell | stimulus | salience | resource |
|---|---|---|---|
| **A** | famous English names ("George Washington") | salient | high |
| **B** | Assyrian ruler names in English ("Ashurbanipal") | obscure | high |
| **B′** | whole fragments, faithful English gloss | obscure | glossed |
| **C** | whole fragments, raw Akkadian transliteration | obscure | low |

**Three read-outs.** Each design renders three times, same code, different table:
`*.png` = ridge everywhere · `*__deck.png` = the per-cell probe the thesis reports
(PLS on fragments, PLS-5 on obscure entities, ridge on A) · `*__pls.png` = best-k PLS
wherever a sweep exists. Use `__deck` for anything compared against the thesis.

---

## Part 1 — the six designs (× 3 read-outs each)

### 1. `slopegraph` — the ladder
**The lead figure.** Four stages left→right (A → B → B′ → C). y is **Δρ against each
arm's own random-init twin in the identical configuration**, not raw ρ — because the
protocols differ per stage, absolute ρ is not comparable but "how far past your own
untrained self" is. The message: the gap decays monotonically, **+.254 → +.217 → +.141
→ +.027**, and TF-IDF (black dashed) falls off the bottom to −.30 at raw Akkadian.
Marker shape encodes which pooling won; dashed grey are random-init Llama twins.

> *Linear time-decoding above a matched untrained control decays monotonically as the
> stimulus becomes more obscure and lower-resource, and reaches zero on raw Akkadian.*

### 2. `heatmap` — every cell at once
The full matrix: rows are the 15 arms, columns are cell × pooling, **ridge panel left,
PLS panel right**. Cell text is raw ρ; cell colour is ρ minus the random-init Qwen
reference in the same column; black frame marks the best arm per column. This is the
only figure that shows both probes simultaneously, and it is where you can see them
disagree — TF-IDF is bright red (top) in the ridge panel and absent/negative in the PLS
panel. No `__deck`/`__pls` variants, because it *is* the comparison.

### 3. `ridgeline` — the distribution, not just the winner
One ridge per configuration (10 total, entity on top, fragment below the labelled
CLIFF divider), each a **discrete 0.05-wide histogram** over the 14 model arms — one
bar-step is literally one arm, so heights are comparable across ridges. Per-arm markers
sit on the baseline; TF-IDF is the black diamond, traced across ridges by a dotted line.
Shows that the fragment cells aren't just lower, they're *compressed* — every arm piles
into the same narrow band.

### 4. `dumbbell` — trained vs its own twin
One row per arm × configuration. Filled dot = trained, open dot = the matched
random-init twin, and the connecting bar is the gap: **green when trained wins, red when
it loses**. Grouped into entity block (teal) and document block (warm). The red bars all
live in the fragment cells — this is the cliff at per-arm granularity, and it's where you
can see Llama-2-70B-random *beating* several trained arms on raw Akkadian.

### 5. `anatomy` — what the probe actually reads
The explainer slide. Shows the literal stimulus text for each cell with the pooled span
underlined and an arrow onto the exact token the probe reads, next to the score it gets.
Use this before any results slide — it prevents the "wait, what is it being shown?"
question that otherwise derails the rest.

### 6. `geometry` — where the gradient lives geometrically
2 × 4 grid of embeddings at each cell's best layer, coloured by chronological rank.
**Top row = supervised PLS-2D, bottom row = unsupervised UMAP.** The point is the
contrast: at entity level both show a clean time gradient; at fragment level PLS can
still forge a weak axis (it is told the answer) while UMAP shows no intrinsic temporal
structure at all. That gap between the two rows is the visual statement of the cliff.

### 7. `kprofile` — how many latent dimensions the date lives in *(new)*
ρ as a function of PLS rank k, one line per arm, per fragment cell × pooling. The ring
marks the selected k; the warm dashed vertical is the **k = 5 ceiling inherited from
`shared/mc_probe.py`**. Several curves are still *climbing* at that line, and 18 of 58
cells selected exactly k = 5 — the signature of a grid that is binding rather than a
fitted choice. This is the figure that justifies the 1–64 sweep, and it redraws itself
with 11 k values once `WAk_pls_ksweep` lands.

---

## Part 2 — layer sweeps and PLS dimension curves

| figure | what it shows |
|---|---|
| `02_cellA_layers` | ρ vs layer depth, cell A. Salient English entities peak **late** (Llama-2-70B at layer 53) — the classic Gurnee–Tegmark profile, and the proof our harness reproduces theirs. |
| `03_cellA_pls` | ρ vs PLS components, cell A. Saturates after very few components: a salient entity's date is a **low-dimensional, nearly linear** direction. |
| `04_cellB_entity_layers` | Same layer sweep for obscure Assyrian ruler names. Peak is lower and **earlier** than cell A — obscure entities are resolved shallower, consistent with less-consolidated knowledge. |
| `05_cellB_entity_pls_bare` | PLS curve, ruler name alone ("Ashurbanipal"). The paper-faithful probe. |
| `06_cellB_entity_pls_all` | PLS curve, name inside a carrier sentence. Compare with 05: context adds little, so the signal is genuinely in the **name token**, not the sentence. |
| `07_fragment_layers` | Layer sweep for whole fragments. Flat and low across depth for every arm — there is no layer where document-level chronology suddenly appears. |
| `08_fragment_pls` | PLS curve for fragments. Contrast with 03: no clean saturation, and the encoders separate from the general LLMs only marginally. |

## Part 3 — embedding panels

| figure | what it shows |
|---|---|
| `09_cellA_sixpanel_PLS` / `10_cellA_sixpanel_UMAP` | Six arms side by side, cell A. In PLS the time gradient is obvious; in UMAP it survives too — the structure is **intrinsic**, not an artifact of supervision. |
| `11_cellB_rulers_sixpanel_PLS` / `12_..._UMAP` | Same for Assyrian ruler names. Gradient is weaker but present in PLS; UMAP is noticeably noisier — the honest picture of an obscure entity. |
| `13_cellB_places_sixpanel_PLS` | Mesopotamian place names — the *space* analog of the ruler panel. |
| `14_cellA_worldplace_sixpanel_PLS` | World places. The strongest geometry anywhere in the study, and the reference point for what a real learned coordinate looks like. |

## Part 4 — manifold geometry (Modell et al., arXiv 2505.18235)

| figure | what it shows |
|---|---|
| `19_isometry_summary` | **The one for the deck.** All 332 full-activation runs condensed. Two panels: **ρ** (kNN-graph geodesic distance vs feature distance) and **ξ** (Chatterjee, cosine vs squared feature distance). Filled dot = trained arm, open dot below = its random-init twin. The finding is the *difference between the panels*: **ρ separates trained from random** (world places .290 vs .101), **ξ does not** — at fragment level Llama-2-70B gets .556 against .463 for its own untrained twin, and an untrained Qwen on the English gloss reaches .531. **So ξ must not be quoted as evidence of a world model**; it is measuring the shape of the activation cloud. Quote ρ, against the matched control. |
| `17_arc__akk__llama2_13b__akk_maximal__year__mean` | Single-cell detail: PCA pairs of the L2-normalised, rank-4-denoised representation coloured by year. Their structure often lives in **PC3×PC4**, not PC1×PC2, which is why four pairs are drawn rather than one. |
| `18_isometry__akk__...` | The two diagnostic scatters for that same cell — cosine vs squared Δyear (ξ annotated) and graph-geodesic vs Δyear (ρ annotated). Use it to show *how* ξ and ρ are computed before showing 19. |
| `15_reducibility_indices` | Engels et al. reducibility: the **ε-mixture index M** (is this direction a mixture of independent 1-D features?) and the **separability index S** (minimum mutual information over rotations). Low M + low S = a genuinely irreducible multi-dimensional feature rather than two stacked scalars. |
| `16_year_metric_choice` | Sanity check on the feature metric itself: absolute Δyear vs their log-recency reparameterisation. Confirms the isometry conclusions aren't an artifact of how "temporal distance" was defined. |

---

## Caveats to keep attached to these figures

1. **Every PLS number predates the k-sweep.** `WAk_pls_ksweep` (job 20110) and
   `WBk_entity_ksweep` (20111) are running; `kprofile` shows why they were needed.
2. **`pls_best_k` is selected on the outer test folds** — the deck's own convention, but
   optimistic. `pls_nested_spearman_mean` (k chosen inside the training rulers) is the
   honest number and is what these figures will quote after the sweep.
3. **R² is not reported for fragment cells.** Under GroupKFold a test fold is
   essentially one ruler, so its year variance is ≈0 and R² degenerates to −0.22 for
   every arm including the floor.
4. **Cell C is weak, not clean, evidence.** Llama-2-70B-random scores .322 there, third
   overall.

## Rebuild

    cd figures && python3 build_tidy.py --mode mc_group --readout deck
    cd designs && for f in ridgeline slopegraph heatmap dumbbell anatomy geometry kprofile; do
        python3 $f.py                                                          # ridge
        TIDY_CSV=../TIDY_all_year_results__mc_group__deck.csv FIG_TAG=__deck python3 $f.py
        TIDY_CSV=../TIDY_all_year_results__mc_group__pls.csv  FIG_TAG=__pls  python3 $f.py
    done
    cd .. && python3 make_curves.py --which all && python3 make_master_fig.py
    cd ../manifold && python3 make_isometry_summary.py
